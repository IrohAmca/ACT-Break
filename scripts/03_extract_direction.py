import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.activation_collector import load_activations
from src.direction_extractor import (
    compare_directions,
    compute_projections,
    extract_direction_from_difference,
    extract_direction_from_probe,
    save_direction,
    visualize_pca,
    visualize_projections,
)
from src.probe_trainer import load_probes

def main():
    print("=" * 60)
    print("ACT-Break — Step 3: Direction Extraction & Validation")
    print("=" * 60)

    activations_data = load_activations(str(config.ACTIVATIONS_DIR / "activations.pt"))
    probe_results = load_probes(str(config.PROBES_DIR / "probe_results.pt"))

    best_layer = probe_results["best_layer"]
    best_probe = probe_results["probes"][best_layer]

    print(f"\nOptimal Layer Selected: L{best_layer}")

    dir_probe = extract_direction_from_probe(best_probe, best_layer)
    dir_diff = extract_direction_from_difference(activations_data, best_layer)

    cosine_sim = compare_directions(dir_probe, dir_diff)
    compute_projections(activations_data, dir_probe["direction"], best_layer)

    pca_path = str(config.FIGURES_DIR / f"pca_L{best_layer}.png")
    visualize_pca(activations_data, best_layer, save_path=pca_path)

    proj_path = str(config.FIGURES_DIR / f"projections_L{best_layer}.png")
    visualize_projections(activations_data, dir_probe["direction"], best_layer, save_path=proj_path)

    save_direction(dir_probe, str(config.OUTPUT_DIR / "direction_probe.pt"))
    save_direction(dir_diff, str(config.OUTPUT_DIR / "direction_diff.pt"))

    print("\n" + "=" * 60)
    print(f"[+] Step 3 complete. Direction cosine similarity: {cosine_sim:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
