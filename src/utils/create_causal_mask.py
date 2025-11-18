import torch


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Causal mask of shape (1, seq_len, seq_len)
    """

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)
