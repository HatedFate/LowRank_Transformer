import torch.nn as nn
from math import sqrt

"""
Simple Implementation of Low Rank Linear Module
"""

class LowRankLinear(nn.Module):

    def __init__(self, in_features, out_features, rank, bias=True):
        """
        Low Rank Modules: Assumes matrix M can be decomposed into AB

        Args:
            in_features: input dim (d_model)
            out_features: output dim (e.g. n_heads * d_k)
            rank: low-rank dimension, must be < min(in_features, out_features)
        """

        super().__init__()
        assert rank <= min(in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.A = nn.Linear(in_features=in_features, out_features=rank, bias=False)
        self.B = nn.Linear(in_features=rank, out_features=out_features, bias=bias)

        # Initialize like LoRA (A random, B near-zero)
        nn.init.kaiming_uniform_(self.A.weight, a=sqrt(5))
        nn.init.zeros_(self.B.weight)

        if bias and self.B.bias is not None:
            nn.init.zeros_(self.B.bias)

    def forward(self, x):
        """
        Forward Module: Q = AB
        """
        
        return self.B(self.A(x))
