import torch
import time

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cpu":
        torch.cpu.synchronize()

def benchmark(device, size=10000, steps=10):
    print(f"\nRunning on {device}...")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up (important for fair timing)
    for _ in range(3):
        torch.matmul(a, b)

    synchronize(device)

    start = time.time()
    for _ in range(steps):
        c = torch.matmul(a, b)
    synchronize(device)
    end = time.time()

    print(f"Time per step: {(end - start) / steps:.4f} seconds")

# CPU
benchmark(torch.device("cpu"))

# Apple GPU (MPS) if available
if torch.backends.mps.is_available():
    benchmark(torch.device("mps"))
else:
    print("\nMPS not available on this system.")

# Intel GPU (XPU) if available
if torch.xpu.is_available():
    benchmark(torch.device("xpu"))
else:
    print("\nIntel GPU (XPU) not available on this system.")

