import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
import re


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)

    value = value.strip().lower()
    hyphen_range = re.fullmatch(r"(\d+)\s*-\s*(\d+)", value)
    if hyphen_range:
        start, end = (int(part) for part in hyphen_range.groups())
        if end < start:
            raise ValueError(f"Invalid descending range for ACT_BREAK_TARGET_LAYERS: {value!r}")
        return list(range(start, end + 1))

    colon_range = re.fullmatch(r"(\d+)\s*:\s*(\d+)", value)
    if colon_range:
        start, end = (int(part) for part in colon_range.groups())
        if end <= start:
            raise ValueError(f"Invalid empty range for ACT_BREAK_TARGET_LAYERS: {value!r}")
        return list(range(start, end))

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str | None, default: list[float | int]) -> list[float | int]:
    if not value:
        return list(default)
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) if item.is_integer() else item for item in parsed]


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    delimiter = "||" if "||" in value else ","
    parsed = [part.strip() for part in value.split(delimiter) if part.strip()]
    return parsed or list(default)


MODEL_PROFILES = {
    "gemma": {
        "model_name": "google/gemma-3-1b-it",
        "target_layers": list(range(8, 18)),
        "advbench_language": "en",
        "negative_activation_mode": "generated",
        "steering_alphas": [0, 1, 2, 5, 10, 15, 20, 30, 50],
    },
    "kara-kumru": {
        "model_name": "AlicanKiraz0/Kara-Kumru-v1.0-2B",
        "target_layers": list(range(6, 16)),
        "advbench_language": "tr",
        "negative_activation_mode": "forced_refusal",
        "steering_alphas": [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5],
    },
    "kumru": {
        "model_name": "vngrs-ai/Kumru-2B",
        "target_layers": list(range(6, 16)),
        "advbench_language": "tr",
        "negative_activation_mode": "forced_refusal",
        "steering_alphas": [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5],
    },
    "kumru-base": {
        "model_name": "vngrs-ai/Kumru-2B-Base",
        "target_layers": list(range(6, 16)),
        "advbench_language": "tr",
        "negative_activation_mode": "forced_refusal",
        "steering_alphas": [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5],
    },
}


def _resolve_profile_key() -> str:
    explicit_profile = os.getenv("ACT_BREAK_MODEL_PROFILE")
    if explicit_profile:
        profile_key = explicit_profile.lower()
        if profile_key not in MODEL_PROFILES:
            choices = ", ".join(sorted(MODEL_PROFILES))
            raise ValueError(f"Unknown ACT_BREAK_MODEL_PROFILE={explicit_profile!r}. Choices: {choices}")
        return profile_key

    requested_model = os.getenv("ACT_BREAK_MODEL_NAME")
    if requested_model:
        for key, profile in MODEL_PROFILES.items():
            if requested_model == profile["model_name"]:
                return key

    return "gemma"


MODEL_PROFILE = _resolve_profile_key()
_MODEL_PROFILE_DATA = MODEL_PROFILES[MODEL_PROFILE]

MODEL_NAME = os.getenv("ACT_BREAK_MODEL_NAME", _MODEL_PROFILE_DATA["model_name"])
MODEL_SUFFIX = os.getenv("ACT_BREAK_MODEL_SUFFIX", MODEL_NAME.split("/")[-1])

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

DTYPE = os.getenv("ACT_BREAK_DTYPE", "float16")
DEVICE = os.getenv("ACT_BREAK_DEVICE", "cuda")

TARGET_LAYERS = _parse_int_list(
    os.getenv("ACT_BREAK_TARGET_LAYERS"),
    _MODEL_PROFILE_DATA["target_layers"],
)

ADVBENCH_LANGUAGE = os.getenv(
    "ACT_BREAK_ADVBENCH_LANGUAGE",
    _MODEL_PROFILE_DATA["advbench_language"],
).lower()

if ADVBENCH_LANGUAGE == "tr":
    COMPLIANCE_PREFIXES = [
        "Elbette, işte",
        "Tabii, işte",
        "Memnuniyetle, işte",
    ]
    REFUSAL_PREFIXES = [
        "Üzgünüm, bu konuda yardımcı olamam.",
        "Buna yardımcı olamam.",
        "Bu isteği yerine getiremem.",
    ]
else:
    COMPLIANCE_PREFIXES = [
        "Sure, here is",
        "Of course! Here's",
        "Absolutely, I can help with that. Here is",
    ]
    REFUSAL_PREFIXES = [
        "I'm sorry, but I can't help with that.",
        "I can't assist with that request.",
        "I'm unable to help with that.",
    ]
DEFAULT_COMPLIANCE_PREFIX = COMPLIANCE_PREFIXES[0]
DEFAULT_REFUSAL_PREFIX = REFUSAL_PREFIXES[0]
NEGATIVE_ACTIVATION_MODE = os.getenv(
    "ACT_BREAK_NEGATIVE_ACTIVATION_MODE",
    _MODEL_PROFILE_DATA["negative_activation_mode"],
).lower()

MAX_PROMPTS = _env_int("ACT_BREAK_MAX_PROMPTS", None)
MAX_NEW_TOKENS = _env_int("ACT_BREAK_MAX_NEW_TOKENS", 20)
BATCH_SIZE = _env_int("ACT_BREAK_BATCH_SIZE", 1)

TEST_SPLIT = 0.2
RANDOM_SEED = 42
PROBE_MAX_ITER = 1000
PROBE_REGULARIZATION = 1.0

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
ADVBENCH_ENGLISH_PATH = DATA_DIR / "harmful_prompts.csv"
ADVBENCH_TURKISH_PATH = DATA_DIR / "harmful_prompts_tr.csv"
ADVBENCH_PATH = ADVBENCH_TURKISH_PATH if ADVBENCH_LANGUAGE == "tr" else ADVBENCH_ENGLISH_PATH

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
GCG_MAX_COORDINATE_UPDATES = _env_int("ACT_BREAK_GCG_MAX_COORDINATE_UPDATES", 3)
GCG_DEDUPLICATE_CANDIDATES = _env_bool("ACT_BREAK_GCG_DEDUPLICATE_CANDIDATES", True)
GCG_REQUIRE_BEHAVIOR_FOR_EARLY_STOP = _env_bool("ACT_BREAK_GCG_REQUIRE_BEHAVIOR_FOR_EARLY_STOP", True)
GCG_ARCHIVE_TOP_N = _env_int("ACT_BREAK_GCG_ARCHIVE_TOP_N", 3)
GCG_ARCHIVE_LIMIT = _env_int("ACT_BREAK_GCG_ARCHIVE_LIMIT", 120)
GCG_TOKEN_DISTANCE_WEIGHT = _env_float("ACT_BREAK_GCG_TOKEN_DISTANCE_WEIGHT", 0.0)
GCG_MOMENTUM = _env_float("ACT_BREAK_GCG_MOMENTUM", 0.9)
GCG_ENABLE_PRUNE_REFINE = _env_bool("ACT_BREAK_GCG_ENABLE_PRUNE_REFINE", True)
GCG_PRUNE_FRACTION = _env_float("ACT_BREAK_GCG_PRUNE_FRACTION", 0.25)
GCG_PRUNE_MIN_TOKENS = _env_int("ACT_BREAK_GCG_PRUNE_MIN_TOKENS", 12)
GCG_PRUNE_REFINE_STEPS = _env_int("ACT_BREAK_GCG_PRUNE_REFINE_STEPS", 25)
GCG_PRUNE_MAX_REL_LOSS_INCREASE = _env_float("ACT_BREAK_GCG_PRUNE_MAX_REL_LOSS_INCREASE", 0.03)

# Loss Weights
LOSS_ALPHA = 1.0                # CE (target token) loss weight
LOSS_BETA = 0.05                # Activation projection loss weight

# Target
TARGET_STRING = os.getenv("ACT_BREAK_TARGET_STRING", DEFAULT_COMPLIANCE_PREFIX)
TARGET_STRINGS = _parse_str_list(
    os.getenv("ACT_BREAK_TARGET_STRINGS"),
    [TARGET_STRING, *[prefix for prefix in COMPLIANCE_PREFIXES if prefix != TARGET_STRING]],
)

# Steering Validation
STEERING_ALPHAS = _parse_float_list(
    os.getenv("ACT_BREAK_STEERING_ALPHAS"),
    _MODEL_PROFILE_DATA["steering_alphas"],
)
STEERING_NUM_PROMPTS = _env_int("ACT_BREAK_STEERING_NUM_PROMPTS", 15)

# Optimization
OPT_NUM_PROMPTS = _env_int("ACT_BREAK_OPT_NUM_PROMPTS", 10)

# Paths
DIRECTION_PATH = OUTPUT_DIR / "direction_probe.pt"
MULTI_LAYER_DIRECTIONS_PATH = OUTPUT_DIR / "directions_multi.pt"
