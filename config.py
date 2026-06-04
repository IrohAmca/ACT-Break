from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
PROBES_DIR = OUTPUT_DIR / "probes"
FIGURES_DIR = OUTPUT_DIR / "figures"
STEERING_DIR = OUTPUT_DIR / "steering"
OPTIMIZATION_DIR = OUTPUT_DIR / "optimization"

for d in [DATA_DIR, ACTIVATIONS_DIR, PROBES_DIR, FIGURES_DIR, STEERING_DIR, OPTIMIZATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = "float16"
DEVICE = "cuda"

TARGET_LAYERS = list(range(12, 19))  # Middle-to-late layers for 24-layer Qwen-0.5B

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
GCG_BATCH_SIZE = 64             # Candidates per iteration (VRAM budget 4GB)
GCG_MINI_BATCH = 16             # Batch evaluation mini-batch size
GCG_MAX_STEPS = 150             # Max iterations (successes converge in ~25 steps)
GCG_CHECK_INTERVAL = 25         # Steps between success checks

# Loss Weights
LOSS_ALPHA = 1.0                # CE (target token) loss weight
LOSS_BETA = 0.3                 # Activation projection loss weight

# Target
TARGET_STRING = "Sure, here is"

# Steering Validation
STEERING_ALPHAS = [0, 1, 2, 5, 10, 15, 20, 30, 50]
STEERING_NUM_PROMPTS = 15

# Optimization
OPT_NUM_PROMPTS = 10            # Number of prompts to optimize

# Paths
DIRECTION_PATH = OUTPUT_DIR / "direction_probe.pt"

