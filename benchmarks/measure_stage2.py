
import torch
import time
import numpy as np
from model.build import EfficientViT_M2_CIFAR

def benchmark_stage2(parallel=False, num_runs=1000, warmup=100):
    print(f"Benchmarking Stage 2 (Parallel={parallel})...")
    
    # Configure parallel stages
    # Stage 2 corresponds to index 1 (0-indexed)
    parallel_stages = [1] if parallel else None
    
    # Instantiate model
    model = EfficientViT_M2_CIFAR(parallel_stages=parallel_stages)
    model.eval()
    
    # Stage 2 is self.blocks2
    stage2_module = model.blocks2
    
    # Input shape for Stage 2
    # Based on analysis: Batch=1, Channels=128, Height=16, Width=16
    # (M2_CIFAR embed_dim[0]=128)
    input_tensor = torch.randn(1, 128, 16, 16)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = stage2_module(input_tensor)
            
    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = stage2_module(input_tensor)
            end = time.time()
            times.append(end - start)
            
    avg_time = np.mean(times) * 1000 # ms
    std_time = np.std(times) * 1000 # ms
    
    print(f"Stage 2 (Parallel={parallel}) Average Time: {avg_time:.4f} ms +/- {std_time:.4f} ms")
    return avg_time

if __name__ == "__main__":
    # Measure Serial
    t_serial = benchmark_stage2(parallel=False)
    print("-" * 30)
    
    # Measure Parallel
    t_parallel = benchmark_stage2(parallel=True)
    print("-" * 30)
    
    print(f"Result Summary:")
    print(f"Serial Stage 2: {t_serial:.4f} ms")
    print(f"Parallel Stage 2: {t_parallel:.4f} ms")
    
    if t_parallel < t_serial:
        print(f"Parallel is {t_serial / t_parallel:.2f}x faster")
    else:
        print(f"Serial is {t_parallel / t_serial:.2f}x faster")
