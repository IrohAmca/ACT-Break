from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
PROBES_DIR = OUTPUT_DIR / "probes"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [DATA_DIR, ACTIVATIONS_DIR, PROBES_DIR, FIGURES_DIR]:
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
