import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def train_probes(activations_data: dict, test_split: float = 0.2,
                 random_seed: int = 42, max_iter: int = 1000,
                 regularization: float = 1.0) -> dict:
    labels = activations_data["labels"].numpy()
    layer_activations = activations_data["activations"]

    probes = {}
    metrics = {}

    print("\n" + "="*50)
    print("Training Linear Probes")
    print("="*50 + "\n")
    print(f"{'Layer':>8} | {'Accuracy':>10} | {'F1':>8} | {'AUC':>8}")
    print("-" * 42)

    for layer_idx in sorted(layer_activations.keys()):
        X = layer_activations[layer_idx].numpy().astype(np.float32)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed, stratify=y
        )

        probe = LogisticRegression(
            max_iter=max_iter,
            C=regularization,
            solver="lbfgs",
            random_state=random_seed,
        )
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        probes[layer_idx] = probe
        metrics[layer_idx] = {"accuracy": acc, "f1": f1, "auc": auc}
        print(f"  L{layer_idx:>4}   | {acc:>9.1%} | {f1:>7.3f} | {auc:>7.3f}")

    best_layer = max(metrics, key=lambda k: metrics[k]["auc"])
    print(f"\n[+] Best Layer: L{best_layer} (AUC: {metrics[best_layer]['auc']:.3f})")

    return {
        "probes": probes,
        "metrics": metrics,
        "best_layer": best_layer,
    }

def save_probes(probe_results: dict, output_path: str):
    torch.save(probe_results, output_path)
    print(f"[+] Saved probes to: {output_path}")

def load_probes(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    best_layer = data["best_layer"]
    n_probes = len(data["probes"])
    print(f"[+] Loaded probes from: {path} ({n_probes} layers, best=L{best_layer})")
    return data
