
import torch
from model.efficientvit import EfficientViT
from model.build import EfficientViT_m2_cifar

def test_structure():
    print("Testing parallel_v2 structure...")
    config = EfficientViT_m2_cifar.copy()
    config['window_size'] = [4, 4, 2]
    
    # Instantiate model with parallel_v2
    model = EfficientViT(
        num_classes=10, 
        parallel_stages=[1], 
        parallel_type='parallel_v2', 
        **config
    )
    model.eval()
    
    x = torch.randn(1, 3, 32, 32)
    try:
        output = model(x)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_structure()
