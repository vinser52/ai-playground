import torch
import time

def benchmark(device, size=20000, steps=10):
    print(f"\nRunning on {device}...")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up (important for fair timing)
    for _ in range(3):
        torch.matmul(a, b)

    torch.cuda.synchronize() if device.type == "cuda" else None

    start = time.time()
    for _ in range(steps):
        c = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    print(f"Time per step: {(end - start) / steps:.4f} seconds")

# CPU
benchmark(torch.device("cpu"))

# Apple GPU (MPS) if available
if torch.backends.mps.is_available():
    benchmark(torch.device("mps"))
else:
    print("\nMPS not available on this system.")

