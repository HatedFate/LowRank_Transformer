import torch
import numpy as np
from typing import List


class TrainingMetrics:
    """Track training metrics"""
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.step = 0
    
    def update(self, loss: float, lr: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step += 1
    
    def get_avg_loss(self, last_n: int = 100):
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses[-last_n:])
    

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Get learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, tokenizer, test_prompts: List[str], temperature: float = 0.7):
    """Evaluate model with test prompts"""
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    print("Generating samples from trained model:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        print("-" * 40)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with different temperatures
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids, 
                max_new_tokens=150,
                temperature=temperature
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Temperature {temperature}: {generated_text}")
            print()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
