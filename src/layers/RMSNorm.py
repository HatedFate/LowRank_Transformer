import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        RMS Normalization

        Args:
            hidden_size: The size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, hidden_size)
        """

        rms = torch.sqrt(torch.mean(hidden_states ** 2, dim=-1, keepdim=True))
        normalized = hidden_states / (rms + self.eps)
        return self.scale * normalized
