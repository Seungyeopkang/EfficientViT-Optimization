
import torch
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.efficientvit import EfficientViT
from model.build import EfficientViT_m2_cifar

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate EfficientViT Accuracy')
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--input-size', type=int, default=32, help='input image size')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--parallel-stages', type=int, nargs='+', default=None, help='stages to run in parallel (e.g. 1 for stage 2)')
    parser.add_argument('--parallel-type', type=str, default='parallel', help='parallel attention type')
    return parser.parse_args()

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
    print(f"Loading model EfficientViT_M2_CIFAR...")
    print(f"Configuration: num_classes=10, window_size=[4, 4, 2], parallel_stages={args.parallel_stages}")
    
    config = EfficientViT_m2_cifar.copy()
    # config['window_size'] = [4, 4, 2]
    
    model = EfficientViT(num_classes=10, parallel_stages=args.parallel_stages, parallel_type=args.parallel_type, **config)
    
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
    
    correct = 0
    total = 0
    
    print("Starting evaluation loop...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i % 10 == 0:
                print(f"Processing batch {i}...")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
    
    with open("eval_result.txt", "w") as f:
        f.write(f"{accuracy:.2f}")

if __name__ == "__main__":
    main()
