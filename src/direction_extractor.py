import numpy as np
import torch
from sklearn.decomposition import PCA

def extract_direction_from_probe(probe, layer_idx: int) -> dict:
    w = probe.coef_[0]
    b = probe.intercept_[0]
    w_norm = w / np.linalg.norm(w)

    print(f"[+] Extracted probe direction vector (L{layer_idx})")
    return {
        "direction": torch.tensor(w_norm, dtype=torch.float32),
        "raw_weights": torch.tensor(w, dtype=torch.float32),
        "bias": float(b),
        "layer": layer_idx,
        "method": "probe",
    }

def extract_direction_from_difference(activations_data: dict, layer_idx: int) -> dict:
    acts = activations_data["activations"][layer_idx].numpy()
    labels = activations_data["labels"].numpy()

    refusal_acts = acts[labels == 0]
    compliance_acts = acts[labels == 1]

    mean_refusal = refusal_acts.mean(axis=0)
    mean_compliance = compliance_acts.mean(axis=0)

    diff = mean_compliance - mean_refusal
    diff_norm = diff / np.linalg.norm(diff)

    print(f"[+] Extracted mean difference vector (L{layer_idx})")
    return {
        "direction": torch.tensor(diff_norm, dtype=torch.float32),
        "raw_difference": torch.tensor(diff, dtype=torch.float32),
        "layer": layer_idx,
        "method": "difference",
    }

def compare_directions(dir1: dict, dir2: dict) -> float:
    v1 = dir1["direction"].numpy()
    v2 = dir2["direction"].numpy()

    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"\n[~] Direction Comparison Similarity: {cosine_sim:.4f}")
    return float(cosine_sim)

def compute_projections(activations_data: dict, direction: torch.Tensor, layer_idx: int) -> dict:
    acts = activations_data["activations"][layer_idx].numpy()
    labels = activations_data["labels"].numpy()

    projections = acts @ direction.numpy()
    refusal_proj = projections[labels == 0]
    compliance_proj = projections[labels == 1]

    print(f"\n[#] Projection Stats (L{layer_idx})")
    print(f"    Refusal:    mean={refusal_proj.mean():.4f}, std={refusal_proj.std():.4f}")
    print(f"    Compliance: mean={compliance_proj.mean():.4f}, std={compliance_proj.std():.4f}")

    return {
        "projections": projections.tolist(),
        "labels": labels.tolist(),
    }

def visualize_pca(activations_data: dict, layer_idx: int, save_path: str | None = None):
    import matplotlib.pyplot as plt

    acts = activations_data["activations"][layer_idx].numpy()
    labels = activations_data["labels"].numpy()

    pca = PCA(n_components=2, random_state=42)
    acts_2d = pca.fit_transform(acts)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(acts_2d[labels == 0, 0], acts_2d[labels == 0, 1], c="#e74c3c", alpha=0.6, label="Refusal")
    ax.scatter(acts_2d[labels == 1, 0], acts_2d[labels == 1, 1], c="#3498db", alpha=0.6, label="Compliance")

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title(f"L{layer_idx} Activation Space (PCA)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Saved PCA plot: {save_path}")
    plt.close(fig)

def visualize_projections(activations_data: dict, direction: torch.Tensor, layer_idx: int, save_path: str | None = None):
    import matplotlib.pyplot as plt

    acts = activations_data["activations"][layer_idx].numpy()
    labels = activations_data["labels"].numpy()
    projections = acts @ direction.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(projections[labels == 0], bins=30, alpha=0.6, color="#e74c3c", label="Refusal", density=True)
    ax.hist(projections[labels == 1], bins=30, alpha=0.6, color="#3498db", label="Compliance", density=True)

    ax.set_xlabel("Projection on Jailbreak Vector", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"L{layer_idx} Projection Distribution", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Saved projection plot: {save_path}")
    plt.close(fig)

def save_direction(direction_data: dict, output_path: str):
    torch.save(direction_data, output_path)
    print(f"[+] Saved direction vector to: {output_path}")
