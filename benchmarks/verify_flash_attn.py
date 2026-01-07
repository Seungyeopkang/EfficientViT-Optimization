
import torch
from model.efficientvit import EfficientViT
from model.build import EfficientViT_m2_cifar

def verify_flash_attn():
    print("Verifying Flash Attention implementation...")
    
    # Configuration for M2 CIFAR
    config = EfficientViT_m2_cifar.copy()
    config['window_size'] = [4, 4, 4] # Match typical usage
    
    # Instantiate model with parallel (V1) structure
    print("Initializing EfficientViT (M2, CIFAR) with parallel_type='parallel' (V1)...")
    model = EfficientViT(num_classes=10, parallel_stages=[1], parallel_type='parallel', **config)
    model.eval()
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 32, 32)
    
    print("Running forward pass...")
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise e

    print("Flash Attention verification complete.")

if __name__ == "__main__":
    verify_flash_attn()
