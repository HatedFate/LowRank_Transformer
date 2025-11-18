import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Position-wise feed-forward network
        
        Args:
            hidden_size: Model dimension
            intermediate_size: Hidden dimension of FFN
            activation_fn: Activation function ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
