
import torch
import time
import argparse
import os
from model.build import EfficientViT_M2_CIFAR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Benchmark EfficientViT Stages')
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--input-size', type=int, default=32, help='input image size')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    return parser.parse_args()

def measure_stage(name, module, x, num_warmup=10, num_runs=100):
    # Warmup
    for _ in range(num_warmup):
        _ = module(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_runs):
        output = module(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000 # ms
    return avg_time, output

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Loader (CIFAR-10 Test)
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    print(f"Loading model EfficientViT_M2_CIFAR with num_classes=10 and window_size=[4, 4, 2]...")
    
    # Manually define config to match the checkpoint
    # Checkpoint uses window_size=[4, 4, 2] based on error analysis
    from model.efficientvit import EfficientViT
    from model.build import EfficientViT_m2_cifar
    
    config = EfficientViT_m2_cifar.copy()
    config['window_size'] = [4, 4, 2]
    
    model = EfficientViT(num_classes=10, **config)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Checkpoint structure handling
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Prepare dummy input for benchmarking (or use a batch from loader)
    # Using a fixed batch ensures consistent measurement without dataloader overhead
    input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)
    
    print(f"\nBenchmarking per-stage inference speed (Batch Size: {args.batch_size})...")
    
    with torch.no_grad():
        # 1. Patch Embed
        t_embed, x = measure_stage("Patch Embed", model.patch_embed, input_tensor)
        print(f"Patch Embed: {t_embed:.4f} ms")
        
        # 2. Stage 1 (blocks1)
        t_stage1, x = measure_stage("Stage 1", model.blocks1, x)
        print(f"Stage 1:     {t_stage1:.4f} ms")
        
        # 3. Stage 2 (blocks2)
        t_stage2, x = measure_stage("Stage 2", model.blocks2, x)
        print(f"Stage 2:     {t_stage2:.4f} ms")
        
        # 4. Stage 3 (blocks3)
        t_stage3, x = measure_stage("Stage 3", model.blocks3, x)
        print(f"Stage 3:     {t_stage3:.4f} ms")
        
        # 5. Head (Pooling + Linear)
        # We need to wrap the head logic slightly to match forward
        class HeadWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                if self.model.distillation:
                    x = self.model.head(x), self.model.head_dist(x)
                    if not self.model.training:
                        x = (x[0] + x[1]) / 2
                else:
                    x = self.model.head(x)
                return x
                
        head_module = HeadWrapper(model)
        t_head, x = measure_stage("Head", head_module, x)
        print(f"Head:        {t_head:.4f} ms")
        
        total_time = t_embed + t_stage1 + t_stage2 + t_stage3 + t_head
        print("-" * 30)
        print(f"Total Sum:   {total_time:.4f} ms")

if __name__ == "__main__":
    main()
