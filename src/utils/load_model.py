import torch
import os


def load_model(model, load_path: str):
    """Load model weights"""
    model.load_state_dict(torch.load(os.path.join(load_path, "model.pt")))
    print(f"Model loaded from {load_path}")
