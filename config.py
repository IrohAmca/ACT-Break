import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

MODEL_NAME = "google/gemma-3-1b-it"
MODEL_SUFFIX = MODEL_NAME.split("/")[-1]

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_SUFFIX
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
PROBES_DIR = OUTPUT_DIR / "probes"
FIGURES_DIR = OUTPUT_DIR / "figures"
STEERING_DIR = OUTPUT_DIR / "steering"
OPTIMIZATION_DIR = OUTPUT_DIR / "optimization"
VALIDATION_DIR = OUTPUT_DIR / "validation"

# Ensure data directory exists
DATA_DIR = PROJECT_ROOT / "data"
for d in [DATA_DIR, ACTIVATIONS_DIR, PROBES_DIR, FIGURES_DIR, STEERING_DIR, OPTIMIZATION_DIR, VALIDATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DTYPE = "float16"
DEVICE = "cuda"

TARGET_LAYERS = list(range(8, 18))  # Middle-to-late layers for 18-layer Gemma-3-1B

COMPLIANCE_PREFIXES = [
    "Sure, here is",
    "Of course! Here's",
    "Absolutely, I can help with that. Here is",
]
DEFAULT_COMPLIANCE_PREFIX = COMPLIANCE_PREFIXES[0]

MAX_PROMPTS = None
MAX_NEW_TOKENS = 20
BATCH_SIZE = 1

TEST_SPLIT = 0.2
RANDOM_SEED = 42
PROBE_MAX_ITER = 1000
PROBE_REGULARIZATION = 1.0

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
ADVBENCH_PATH = DATA_DIR / "harmful_prompts.csv"

# === Module 2: Jailbreak Suffix Discovery ===

# GCG Hyperparameters
SUFFIX_LENGTH = 20              # Suffix token count
SUFFIX_INIT_TOKEN = "!"         # Suffix initial token
GCG_TOP_K = 256                 # Candidate tokens per position

# Default GCG parameters (safe fallback for small/local GPUs like GTX 1650 4GB)
GCG_BATCH_SIZE = 64
GCG_MINI_BATCH = 16

# Try to detect GPU VRAM and scale up GCG parameters for high-end GPUs (e.g. A100 on Colab)
try:
    import torch
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_mem
        vram_gb = vram_bytes / (1024 ** 3)
        if vram_gb >= 35:    # A100 (40GB/80GB), H100, etc.
            GCG_BATCH_SIZE = 512
            GCG_MINI_BATCH = 128
        elif vram_gb >= 12:  # T4 (16GB), L4 (24GB), RTX 3090/4090, etc.
            GCG_BATCH_SIZE = 256
            GCG_MINI_BATCH = 64
except Exception:
    pass

GCG_MAX_STEPS = 150             # Max iterations (successes converge in ~25 steps)
GCG_CHECK_INTERVAL = 25         # Steps between success checks

# Loss Weights
LOSS_ALPHA = 1.0                # CE (target token) loss weight
LOSS_BETA = 0.05                # Activation projection loss weight

# Target
TARGET_STRING = "Sure, here is"

# Steering Validation
STEERING_ALPHAS = [0, 1, 2, 5, 10, 15, 20, 30, 50]
STEERING_NUM_PROMPTS = 15

# Optimization
OPT_NUM_PROMPTS = 10            # Number of prompts to optimize

# Paths
DIRECTION_PATH = OUTPUT_DIR / "direction_probe.pt"
MULTI_LAYER_DIRECTIONS_PATH = OUTPUT_DIR / "directions_multi.pt"

