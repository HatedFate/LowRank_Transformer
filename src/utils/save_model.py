import torch
import os


def save_model(model, tokenizer, save_path: str):
    """Save model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    if hasattr(model, 'config'):
        torch.save(model.config.__dict__, os.path.join(save_path, "config.json"))
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
