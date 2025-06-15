import torch
import torch.nn as nn
import torch.optim as optim

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # 1) MLP backbone with optional dropout
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers).to(self.device)

        # 2) Single deterministic output layer
        self.out = nn.Linear(in_dim, 1).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: returns the point mean prediction.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        h = self.backbone(x)
        return self.out(h).squeeze(-1)

    def fit(
        self,
        train_seq,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        """
        train_seq: iterable of (seq, label) where seq is shape [input_size] or [batch, input_size]
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # wrap into DataLoader if needed
        if not isinstance(train_seq, torch.utils.data.DataLoader):
            ds = []
            for seq, label in train_seq:
                x = torch.tensor(seq, dtype=torch.float32)
                y = torch.tensor(label, dtype=torch.float32).view(-1, 1)
                ds.append((x, y))
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        else:
            loader = train_seq

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                preds = self(x_batch)
                loss = loss_fn(preds, y_batch.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            if epoch % 10 == 0:
                print(f"[fit] Epoch {epoch:>3}, MSE = {avg:.4f}")

    def predict(
        self,
        horizon: int,
        last_input: list[float],
    ) -> list[float]:
        """
        Recursive forecast for `horizon` steps using a simple deterministic pass.
        Returns:
          - preds: list of length=horizon
        """
        self.eval()  # disable dropout
        preds = []
        seq = last_input.copy()
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                y = self(x).item()
                preds.append(y)
                seq = seq[1:] + [y]
        return preds
    
    def reset(self):
        """
        Reset the model state. This is a no-op for MLP, but can be overridden in subclasses.
        """
        pass

    def get_num_params(self) -> int:
        """
        Returns the total number of parameters in the MLP model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params(self) -> dict:
        """
        Returns a dictionary of model parameters.
        """
        return {
            'input_size': self.backbone[0].in_features,
            'hidden_sizes': [layer.out_features for layer in self.backbone if isinstance(layer, nn.Linear)],
            'dropout': self.backbone[2].p if len(self.backbone) > 2 and isinstance(self.backbone[2], nn.Dropout) else 0.0
        }
