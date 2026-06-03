import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.download_advbench import download_advbench
from src.activation_collector import collect_activations, load_prompts, save_activations
from src.model_loader import HookedModel

def main():
    print("=" * 60)
    print("ACT-Break — Step 1: Collect Contrastive Activations")
    print("=" * 60)

    download_advbench()
    prompts = load_prompts(str(config.ADVBENCH_PATH), max_prompts=config.MAX_PROMPTS)

    model = HookedModel(
        model_name=config.MODEL_NAME,
        target_layers=config.TARGET_LAYERS,
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    model.load()

    result = collect_activations(
        hooked_model=model,
        prompts=prompts,
        target_layers=config.TARGET_LAYERS,
        compliance_prefix=config.DEFAULT_COMPLIANCE_PREFIX,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    output_path = str(config.ACTIVATIONS_DIR / "activations.pt")
    save_activations(result, output_path)

    model.remove_hooks()
    print(f"\n[+] Step 1 finished. Activations saved to {output_path}")

if __name__ == "__main__":
    main()
