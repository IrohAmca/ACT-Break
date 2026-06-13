import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

import config


def load_tokenizer_metadata(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
    except ValueError as exc:
        message = str(exc)
        if "Tokenizer class" not in message or "does not exist" not in message:
            raise
        print("[WARN] AutoTokenizer could not resolve tokenizer class; trying PreTrainedTokenizerFast.")
        return PreTrainedTokenizerFast.from_pretrained(model_name, padding_side="left")


def main():
    print("=" * 60)
    print("ACT-Break -- Model Configuration Check")
    print("=" * 60)

    hf_config = AutoConfig.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    num_layers = getattr(hf_config, "num_hidden_layers", None) or getattr(hf_config, "n_layer", None)
    if num_layers is None:
        raise ValueError("Could not infer the number of hidden layers from the Hugging Face config.")

    max_layer = num_layers - 1
    invalid_layers = [layer for layer in config.TARGET_LAYERS if layer < 0 or layer > max_layer]
    if invalid_layers:
        raise ValueError(
            f"Configured target layers are invalid: {invalid_layers}. "
            f"{config.MODEL_NAME} has {num_layers} layers, valid range is 0-{max_layer}."
        )

    tokenizer = load_tokenizer_metadata(config.MODEL_NAME)
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    print(f"Profile:                 {config.MODEL_PROFILE}")
    print(f"Model:                   {config.MODEL_NAME}")
    print(f"Output suffix:           {config.MODEL_SUFFIX}")
    print(f"Model type:              {getattr(hf_config, 'model_type', 'unknown')}")
    print(f"Hidden layers:           {num_layers}")
    print(f"Target layers:           {config.TARGET_LAYERS}")
    print(f"Language:                {config.ADVBENCH_LANGUAGE}")
    print(f"Negative activation:     {config.NEGATIVE_ACTIVATION_MODE}")
    print(f"Compliance prefix:       {config.DEFAULT_COMPLIANCE_PREFIX!r}")
    print(f"Refusal prefix:          {config.DEFAULT_REFUSAL_PREFIX!r}")
    print(f"Steering alphas:         {config.STEERING_ALPHAS}")
    print(f"Tokenizer chat template: {'yes' if has_chat_template else 'no'}")

    if not has_chat_template:
        print("[WARN] Tokenizer has no chat template; HookedModel will use a plain User/Assistant fallback.")
    if config.ADVBENCH_LANGUAGE == "tr" and not config.ADVBENCH_TURKISH_PATH.exists():
        print(f"[WARN] Turkish AdvBench file is missing: {config.ADVBENCH_TURKISH_PATH}")
    if config.ADVBENCH_LANGUAGE != "tr" and not config.ADVBENCH_ENGLISH_PATH.exists():
        print(f"[WARN] English AdvBench file is missing: {config.ADVBENCH_ENGLISH_PATH}")

    print("[OK] Model configuration check passed.")


if __name__ == "__main__":
    main()
