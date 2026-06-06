import torch
import config
from src.model_loader import HookedModel

print("=== BASE MODEL MEMORY BENCHMARK ===")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load model
model = HookedModel(
    model_name=config.MODEL_NAME,
    target_layers=[8],
    dtype=config.DTYPE,
    device=config.DEVICE,
)
model.load()

allocated = torch.cuda.memory_allocated() / (1024 * 1024)
max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
reserved = torch.cuda.memory_reserved() / (1024 * 1024)

print(f"\n[+] Memory Allocated: {allocated:.1f} MB")
print(f"[+] Max Memory Allocated during load: {max_allocated:.1f} MB")
print(f"[+] Memory Reserved: {reserved:.1f} MB")
