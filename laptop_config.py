import platform
import os
import psutil
import torch

print("=" * 40)
print("SYSTEM INFORMATION")
print("=" * 40)

print(f"OS            : {platform.system()} {platform.release()}")
print(f"OS Version    : {platform.version()}")
print(f"Architecture  : {platform.machine()}")
print(f"Processor     : {platform.processor()}")

print("\nMEMORY")
mem = psutil.virtual_memory()
print(f"Total RAM     : {mem.total / (1024**3):.2f} GB")
print(f"Available RAM : {mem.available / (1024**3):.2f} GB")

print("\nCPU")
print(f"Physical cores: {psutil.cpu_count(logical=False)}")
print(f"Logical cores : {psutil.cpu_count(logical=True)}")

print("\nPYTHON / TORCH")
print(f"Python        : {platform.python_version()}")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

print("=" * 40)
