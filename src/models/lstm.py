import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional, List

class LSTM(nn.Module):
    def __init__(
        self,
        input_size:   int    = 1,
        hidden_size:  int    = 64,
        num_layers:   int    = 1,
        output_size:  int    = 1,
        dropout:      float  = 0.0,
        device:       str    = 'cpu',
    ):
        super().__init__()
        self.device       = torch.device(device)
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_size  = output_size

        # core LSTM encoder with inter-layer dropout
        self.lstm    = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        # optional dropout on final hidden state
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # single head for point prediction
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.to(self.device)
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shaping: [batch, seq_len, input_size]
        if x.dim() == 1:
            x = x.view(1, -1, self.input_size)
        elif x.dim() == 2 and self.input_size == 1:
            x = x.unsqueeze(-1)
        x = x.to(self.device)

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        h_last = out[:, -1, :]
        h_last = self.dropout(h_last)
        return self.fc_out(h_last)

    def fit(
        self,
        train_data,
        epochs: int = 10,
        lr: float  = 1e-3,
        optimizer_class=optim.Adam,
    ):
        """
        train_data: iterable of (seq, label), seq: [seq_len] or [batch, seq_len]
        """
        if self.optimizer is None:
            self.optimizer = optimizer_class(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for seq, label in train_data:
                x = torch.FloatTensor(seq).to(self.device)
                y = torch.FloatTensor(label).view(-1, 1).to(self.device)

                self.optimizer.zero_grad()
                preds = self(x)
                loss = loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            avg = total_loss / len(train_data)
            if epoch % 10 == 0:
                print(f"[fit] Epoch {epoch:>3}, MSE = {avg:.6f}")

    def predict(
        self,
        history: list,
        horizon: Optional[int] = 1,
    ) -> list[float]:
        """
        Recursive deterministic forecast for `horizon` steps.
        Returns list of length `horizon`.
        """
        self.eval()
        preds = []
        seq = history.copy()
        with torch.no_grad():
            if horizon > 1:
                for _ in range(horizon):
                    x = torch.FloatTensor(seq).to(self.device)
                    y = self(x).item()
                    preds.append(y)
                    seq.pop(0)
                    seq.append(y)

            else:
                x = torch.FloatTensor(seq).to(self.device)
                preds = list(self(x))
        return preds
    
    def get_num_params(self) -> int:
        """
        Returns the number of parameters in the LSTM model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params(self) -> dict:
        """
        Returns the parameters of the LSTM model as a dictionary.
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0
        }
