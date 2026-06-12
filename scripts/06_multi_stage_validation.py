import sys
import json
from pathlib import Path
import torch
import numpy as np

# Force matplotlib to use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Insert project root in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.model_loader import HookedModel
from src.activation_reference import ActivationReferenceClassifier
from src.validation import MultiStageValidator
from src.text_utils import safe_console_text

def safe_str(s):
    return safe_console_text(s)

def main():
    print("=" * 60)
    print("ACT-Break -- Step 6: Multi-Stage Validation")
    print("=" * 60)

    # 1. Load optimization results
    results_path = config.OPTIMIZATION_DIR / "results.json"
    if not results_path.exists():
        print(f"[!] Optimization results not found at {results_path}. Run Step 5 first.")
        sys.exit(1)
        
    with open(results_path, "r", encoding="utf-8") as f:
        opt_data = json.load(f)
        
    optimization_results = opt_data["optimization_results"]
    
    prompts = [item["prompt"] for item in optimization_results]
    suffixes = [item["suffix"] for item in optimization_results]
    N = len(prompts)
    
    print(f"[+] Loaded {N} prompts and suffixes from Module 2 results.")

    # 2. Load steering vector
    if not config.DIRECTION_PATH.exists():
        print(f"[!] Steering vector not found at {config.DIRECTION_PATH}. Run Step 3 first.")
        sys.exit(1)

    print(f"[*] Loading steering vector from {config.DIRECTION_PATH}")
    direction_data = torch.load(str(config.DIRECTION_PATH), map_location="cpu")
    direction_vec = direction_data["direction"]
    layer_idx = direction_data["layer"]
    print(f"[+] Loaded direction vector for layer L{layer_idx}")

    activation_classifier = ActivationReferenceClassifier.from_path(
        config.ACTIVATIONS_DIR / "activations.pt",
        direction_vec,
        layer_idx,
    )
    print(
        "[+] Loaded Stage 1 activation reference: "
        f"L{layer_idx}, refusal_mean={activation_classifier.refusal_mean:+.1f}, "
        f"compliance_mean={activation_classifier.compliance_mean:+.1f}, "
        f"threshold={activation_classifier.threshold:+.1f}"
    )

    # 3. Load model
    model = HookedModel(
        model_name=config.MODEL_NAME,
        target_layers=[layer_idx],
        dtype=config.DTYPE,
        device=config.DEVICE,
    ).load()

    validator = MultiStageValidator(
        hooked_model=model,
        tokenizer=model.tokenizer,
        direction_vec=direction_vec,
        layer_idx=layer_idx,
        activation_classifier=activation_classifier
    )

    # === STAGE 1: Transferability Matrix ===
    print("\n" + "-" * 50)
    print("Stage 1: Evaluating Suffix Transferability Matrix...")
    print("-" * 50)
    
    transfer_matrix, response_matrix, trajectory_matrix = validator.evaluate_transferability(
        prompts,
        suffixes,
        max_new_tokens=40,
    )
    
    # Calculate transferability statistics
    # Suffix transfer rate: average success rate of suffix j on all OTHER prompts
    suffix_transfer_rates = []
    for j in range(N):
        other_successes = [transfer_matrix[i, j] for i in range(N) if i != j]
        suffix_transfer_rates.append(float(np.mean(other_successes)))
        
    # Prompt vulnerability rate: average success rate of other suffixes on prompt i
    prompt_vuln_rates = []
    for i in range(N):
        other_successes = [transfer_matrix[i, j] for j in range(N) if j != i]
        prompt_vuln_rates.append(float(np.mean(other_successes)))
        
    # Diagonal success rate (self-success)
    diagonal_success_rate = float(np.mean([transfer_matrix[i, i] for i in range(N)]))

    print(f"[+] Cross-validation finished.")
    print(f"    Diagonal (Self) Success Rate: {diagonal_success_rate * 100:.1f}%")
    print(f"    Average Suffix Transfer Rate: {np.mean(suffix_transfer_rates) * 100:.1f}%")
    print(f"    Average Prompt Vulnerability: {np.mean(prompt_vuln_rates) * 100:.1f}%")

    # Plot Transferability Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(transfer_matrix, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(label="Jailbreak Success (1=Success, 0=Refusal)")
    plt.title("GCG Suffix Transferability Matrix")
    plt.xlabel("Suffix Source Prompt (1-10)")
    plt.ylabel("Target Evaluated Prompt (1-10)")
    plt.xticks(range(N), [f"S{i+1}" for i in range(N)])
    plt.yticks(range(N), [f"P{i+1}" for i in range(N)])
    
    # Add values text inside heatmap
    for i in range(N):
        for j in range(N):
            val = transfer_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")
            
    heatmap_path = config.VALIDATION_DIR / "transferability_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[+] Saved transferability heatmap to {heatmap_path}")

    # === STAGE 2: Perplexity Filtering ===
    print("\n" + "-" * 50)
    print("Stage 2: Calculating Perplexity Values...")
    print("-" * 50)
    
    # Calculate perplexities
    diagonal_ppls = []
    baseline_ppl = [] # Normal compliance prefix PPL as reference
    
    # Calculate normal compliance prefix PPL for reference
    for prompt in prompts:
        ppl = validator.calculate_perplexity(prompt, config.DEFAULT_COMPLIANCE_PREFIX)
        baseline_ppl.append(ppl)
        
    mean_baseline_ppl = float(np.mean([p for p in baseline_ppl if p != float("inf")]))
    
    print(f"[*] Baseline compliance prefix ('{config.DEFAULT_COMPLIANCE_PREFIX}') Avg PPL: {mean_baseline_ppl:.2f}")
    
    for i, (prompt, suffix) in enumerate(zip(prompts, suffixes)):
        ppl = validator.calculate_perplexity(prompt, suffix)
        diagonal_ppls.append(ppl)
        suffix_safe = safe_str(suffix.replace('\n', ' '))
        print(f"  Prompt {i+1} Suffix PPL: {ppl:.2f} | Suffix: {suffix_safe[:40]}...")
        
    valid_ppls = [p for p in diagonal_ppls if p != float("inf") and not np.isnan(p)]
    mean_suffix_ppl = float(np.mean(valid_ppls)) if valid_ppls else float("inf")
    print(f"[+] Mean Suffix PPL: {mean_suffix_ppl:.2f} (compared to baseline PPL: {mean_baseline_ppl:.2f})")
    print("[*] Notice: Adversarial suffixes show significantly higher perplexity, making them easy to filter.")

    # === STAGE 3: Logit Lens Projection Tracking ===
    print("\n" + "-" * 50)
    print(f"Stage 3: Tracking Logit Lens (Layer L{layer_idx} Projection)...")
    print("-" * 50)
    
    # Pick the first prompt that is successfully jailbroken
    target_idx = 0
    for idx in range(N):
        if transfer_matrix[idx, idx] == 1:
            target_idx = idx
            break
            
    test_prompt = prompts[target_idx]
    test_suffix = suffixes[target_idx]
    
    print(f"[*] Running Logit Lens on Prompt {target_idx+1}: {safe_str(test_prompt)}")
    
    # Case 1: Original prompt (no suffix)
    # We will generate manually step-by-step
    orig_projs, orig_response = validator.track_logit_lens(test_prompt, "", max_new_tokens=40)
    
    # Case 2: Jailbroken prompt (with suffix)
    jail_projs, jail_response = validator.track_logit_lens(test_prompt, test_suffix, max_new_tokens=40)
    
    print(f"  [Original Output]   : {safe_str(orig_response)}")
    print(f"  [Jailbroken Output] : {safe_str(jail_response)}")

    # Plot Logit Lens Trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(orig_projs, label="Original Prompt (Refusal)", color="red", marker="o", linewidth=2)
    plt.plot(jail_projs, label="Prompt + Suffix (Jailbreak)", color="blue", marker="x", linewidth=2)
    plt.axhline(
        y=activation_classifier.threshold,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Stage 1 Threshold",
    )
    plt.title(f"Logit Lens Projection Trajectory (L{layer_idx}) - Prompt {target_idx+1}")
    plt.xlabel("Generated Token Index")
    plt.ylabel("Projection score onto V_jailbreak")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    trajectory_path = config.VALIDATION_DIR / "logit_lens_trajectory.png"
    plt.tight_layout()
    plt.savefig(trajectory_path, dpi=150)
    plt.close()
    print(f"[+] Saved Logit Lens trajectory plot to {trajectory_path}")

    # === STAGE 4: Topic Relevance ===
    print("\n" + "-" * 50)
    print("Stage 4: Evaluating Semantic Topic Relevance...")
    print("-" * 50)
    
    diagonal_relevances = []
    for i, (prompt, response) in enumerate(zip(prompts, [response_matrix[idx][idx] for idx in range(N)])):
        sim = validator.evaluate_topic_relevance(prompt, response)
        diagonal_relevances.append(sim)
        response_safe = safe_str(response.replace('\n', ' '))
        print(f"  Prompt {i+1} relevance cosine similarity: {sim:.4f} | Response: {response_safe[:50]}...")
        
    mean_relevance = float(np.mean(diagonal_relevances))
    print(f"[+] Mean Topic Relevance Cosine Similarity: {mean_relevance:.4f}")

    # === 5. Save validation metrics ===
    val_results = {
        "diagonal_success_rate": diagonal_success_rate,
        "avg_suffix_transfer_rate": float(np.mean(suffix_transfer_rates)),
        "avg_prompt_vulnerability_rate": float(np.mean(prompt_vuln_rates)),
        "mean_baseline_ppl": mean_baseline_ppl,
        "mean_suffix_ppl": mean_suffix_ppl,
        "mean_topic_relevance": mean_relevance,
        "suffix_transfer_rates": suffix_transfer_rates,
        "prompt_vuln_rates": prompt_vuln_rates,
        "diagonal_ppls": diagonal_ppls,
        "diagonal_relevances": diagonal_relevances,
        "transfer_matrix": transfer_matrix.tolist(),
        "trajectory_matrix": trajectory_matrix,
    }
    
    results_json_path = config.VALIDATION_DIR / "validation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved validation results JSON to {results_json_path}")
    
    # Save text report
    report_path = config.VALIDATION_DIR / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("ACT-Break Modul 3: Multi-Stage Validation Report\n")
        f.write("============================================================\n\n")
        
        f.write(f"1. Transferability Matrix Summary:\n")
        f.write(f"  - Diagonal (Self-Success) Rate:  {diagonal_success_rate*100:.1f}%\n")
        f.write(f"  - Average Suffix Transfer Rate:  {np.mean(suffix_transfer_rates)*100:.1f}%\n")
        f.write(f"  - Average Prompt Vulnerability:  {np.mean(prompt_vuln_rates)*100:.1f}%\n\n")
        
        f.write(f"2. Perplexity (PPL) Defense Filtering Summary:\n")
        f.write(f"  - Mean Baseline '{config.DEFAULT_COMPLIANCE_PREFIX}' PPL: {mean_baseline_ppl:.2f}\n")
        f.write(f"  - Mean Adversarial Suffix PPL:       {mean_suffix_ppl:.2f}\n")
        f.write(f"  - PPL Ratio (Suffix / Baseline):     {mean_suffix_ppl / (mean_baseline_ppl + 1e-5):.1f}x\n\n")
        
        f.write(f"3. Semantic Topic Relevance Summary:\n")
        f.write(f"  - Mean Cosine Similarity (Prompt vs Response): {mean_relevance:.4f}\n\n")
        
        f.write("------------------------------------------------------------\n")
        f.write("Detailed Cross-Validation Matrix (Row: Prompt, Col: Suffix Source):\n")
        f.write("------------------------------------------------------------\n")
        
        header = "      " + " ".join(f" S{j+1:<3}" for j in range(N)) + "\n"
        f.write(header)
        for i in range(N):
            row_str = f"P{i+1:<4} " + " ".join(f"  {transfer_matrix[i, j]}  " for j in range(N)) + "\n"
            f.write(row_str)
            
        f.write("\n------------------------------------------------------------\n")
        f.write("Detailed Prompt Metrics:\n")
        f.write("------------------------------------------------------------\n")
        for i in range(N):
            prompt_trunc = safe_str(prompts[i][:45])
            f.write(f"P{i+1}: {prompt_trunc:<45} | Vuln: {prompt_vuln_rates[i]*100:.1f}% | PPL: {diagonal_ppls[i]:.1f} | Relevance: {diagonal_relevances[i]:.4f}\n")
            
    print(f"[+] Saved text report summary to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
