import torch
import torch.nn as nn
import torch.optim as optim

class Conv1D(nn.Module):
    def __init__(
        self,
        input_size:      int,
        conv_channels:   list[int],
        kernel_sizes:    list[int],
        strides:         list[int],
        paddings:        list[int],
        input_length:    int,
        output_seq_len:  int,
        dropout:         float = 0.0,
        device:          str   = "cpu",
    ):
        """
        Args:
          input_size:     # of input channels
          conv_channels:  list of # of output channels for each conv layer
          kernel_sizes:   list of kernel sizes
          strides:        list of strides
          paddings:       list of paddings
          input_length:   length of the 1D input sequence
          output_seq_len: how many future steps to forecast
        """
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "conv_channels, kernel_sizes, strides, paddings must all be same length"

        self.device         = torch.device(device)
        self.input_length   = input_length
        self.output_seq_len = output_seq_len
        self.dropout        = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # build stacked conv layers
        self.convs = nn.ModuleList()
        in_ch = input_size
        curr_len = input_length
        for out_ch, k, s, p in zip(conv_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
            self.convs.append(conv)
            # update length after this conv:
            curr_len = (curr_len + 2*p - k) // s + 1
            in_ch = out_ch

        self.flatten_dim = in_ch * curr_len
        self.fc_out = nn.Linear(self.flatten_dim, output_seq_len)

        self.to(self.device)
        self.optimizer = None

    def forward(self, x):
        """
        x: sequence of shape [seq_len] or Tensor [channels, seq_len]
        returns: Tensor [1, output_seq_len]
        """
        # prepare [batch=1, channels, seq_len]
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)          # [1, seq_len]
        x = x.unsqueeze(0).to(self.device)  # [1, 1, seq_len] if input_size=1
                                            # or [1, channels, seq_len]

        z = x
        for conv in self.convs:
            z = conv(z)
            z = nn.functional.relu(z)

        z = z.flatten(start_dim=1)        # [1, flatten_dim]
        z = self.dropout(z)
        return self.fc_out(z)             # [1, output_seq_len]

    def fit(
        self,
        train_data,
        epochs: int = 50,
        lr: float = 1e-2,
        optimizer_class=optim.Adam,
    ):
        """
        train_data: iterable of (seq, target) pairs
          - seq: list or array length=input_length
          - target: list or array length=output_seq_len
        """
        if self.optimizer is None:
            self.optimizer = optimizer_class(self.parameters(), lr=lr)

        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for seq, target in train_data:
                y = torch.FloatTensor(target).view(1, -1).to(self.device)  # [1, output_seq_len]
                self.optimizer.zero_grad()
                y_pred = self(seq)           # [1, output_seq_len]
                loss   = loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                avg = total_loss / len(train_data)
                print(f"[fit] Epoch {epoch:>3}, MSE = {avg:.6f}")

    def predict(self, last_sequence: list, horizon: int) -> list[float]:
        """
        Recursive point forecasts for `horizon` steps:
          - each step: run forward → [1, output_seq_len], take the first element
          - slide window by one
        """
        self.eval()
        seq = last_sequence.copy()
        forecasts = []

        with torch.no_grad():
            if horizon > 1:
                for _ in range(horizon):
                    out = self(seq).view(-1).cpu().tolist()  # length=output_seq_len
                    next_val = out[0]
                    forecasts.append(next_val)
                    seq.pop(0)
                    seq.append(next_val)
            else:
                # k step ahead forecast
                out = self(seq).view(-1).cpu().tolist()
                forecasts = list(out[:self.output_seq_len])

        return forecasts
    
    def get_num_params(self) -> int:
        """
        Returns the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params(self):
        """
        Returns a dictionary of model parameters.
        """
        return {
            'input_size': self.input_length,
            'output_seq_len': self.output_seq_len,
            'conv_channels': [conv.out_channels for conv in self.convs],
            'kernel_sizes': [conv.kernel_size[0] for conv in self.convs],
            'strides': [conv.stride[0] for conv in self.convs],
            'paddings': [conv.padding[0] for conv in self.convs],
            'dropout': self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0
        }
