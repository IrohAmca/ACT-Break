import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.activation_collector import load_activations
from src.probe_trainer import save_probes, train_probes

def main():
    print("=" * 60)
    print("ACT-Break — Step 2: Linear Probe Training")
    print("=" * 60)

    activations_path = str(config.ACTIVATIONS_DIR / "activations.pt")
    activations_data = load_activations(activations_path)

    probe_results = train_probes(
        activations_data=activations_data,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED,
        max_iter=config.PROBE_MAX_ITER,
        regularization=config.PROBE_REGULARIZATION,
    )

    output_path = str(config.PROBES_DIR / "probe_results.pt")
    save_probes(probe_results, output_path)
    print(f"\n[+] Step 2 finished. Probes saved to {output_path}")

if __name__ == "__main__":
    main()
