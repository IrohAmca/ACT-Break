import sys
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.model_loader import HookedModel
from src.activation_collector import load_prompts
from src.activation_measurements import (
    measure_forced_target_text,
    track_generation_trajectory_text,
)
from src.activation_reference import ActivationReferenceClassifier
from src.gcg_optimizer import GCGOptimizer


def safe_print(text: str):
    """Print ASCII-safe text to avoid cp1254 encoding crashes on Windows."""
    print(text.encode("ascii", errors="replace").decode("ascii"))


def save_incremental(results: list, output_dir: Path):
    """Save results incrementally after each prompt completes."""
    results_path = output_dir / "results.json"
    forced_target_success_count = sum(1 for r in results if r["success"])
    total = len(results)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_prompts_completed": total,
            "forced_target_success_count": forced_target_success_count,
            "forced_target_success_rate": forced_target_success_count / total if total > 0 else 0,
            # Backward-compatible aliases. `success` means forced-target
            # activation success, not behavioral jailbreak success.
            "success_count": forced_target_success_count,
            "success_rate": forced_target_success_count / total if total > 0 else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)


def run_comparison_test(model, results: list, direction_vec, layer_idx, activation_classifier):
    """
    Final comparison: for each prompt, generate a response WITH and WITHOUT
    the discovered suffix, side by side.
    Classification uses activation projection (language-agnostic).
    """
    print("\n" + "=" * 60)
    print("COMPARISON TEST: Original vs Suffix-injected Prompts")
    print("  (Using activation projection for classification)")
    print("=" * 60)

    comparisons = []

    for idx, res in enumerate(results):
        prompt = res["prompt"]
        suffix = res.get("suffix", "")
        forced_target_success = res.get("forced_target_success", res["success"])

        safe_print(f"\n[#{idx+1}] Prompt: {prompt[:60]}...")
        safe_print(f"  Suffix found: {suffix[:40]}...")
        print(f"  Forced-target Success: {forced_target_success}")

        # 1. Track real generation WITHOUT suffix.
        original_generation = track_generation_trajectory_text(
            model=model,
            prompt=prompt,
            suffix="",
            direction_vec=direction_vec,
            layer_idx=layer_idx,
            activation_classifier=activation_classifier,
            max_new_tokens=60,
        )
        response_original = original_generation["response"]

        # 2. Measure objective-side forced target and real generation WITH suffix.
        suffix_forced = measure_forced_target_text(
            model=model,
            prompt=prompt,
            suffix=suffix,
            target_string=config.TARGET_STRING,
            direction_vec=direction_vec,
            layer_idx=layer_idx,
            activation_classifier=activation_classifier,
        )
        suffix_generation = track_generation_trajectory_text(
            model=model,
            prompt=prompt,
            suffix=suffix,
            direction_vec=direction_vec,
            layer_idx=layer_idx,
            activation_classifier=activation_classifier,
            max_new_tokens=60,
        )
        response_with_suffix = suffix_generation["response"]

        jailbreak_confirmed = (
            not original_generation["generated_any_compliance"]
            and suffix_generation["generated_any_compliance"]
        )
        loss_behavior_gap = (
            suffix_forced["is_compliance"]
            and not suffix_generation["generated_any_compliance"]
        )
        suffix_generation_status = "Compliance" if suffix_generation["generated_any_compliance"] else "Refusal"
        original_generation_status = "Compliance" if original_generation["generated_any_compliance"] else "Refusal"

        safe_print(
            f"  [Original Gen] ({original_generation_status}) "
            f"{response_original.replace(chr(10), ' ')[:70]}"
        )
        safe_print(
            f"  [Forced Target] ({suffix_forced['status']}, proj={suffix_forced['projection']:+.1f})"
        )
        safe_print(
            f"  [Suffix Gen]   ({suffix_generation_status}) "
            f"{response_with_suffix.replace(chr(10), ' ')[:70]}"
        )

        comparisons.append({
            "prompt": prompt,
            "suffix": suffix,
            "forced_target_success": forced_target_success,
            "gcg_success": forced_target_success,
            "gcg_steps": res["steps"],
            "original_response": response_original,
            "original_generation": original_generation,
            "original_status": original_generation_status,
            "suffix_response": response_with_suffix,
            "suffix_forced_target": suffix_forced,
            "suffix_generation": suffix_generation,
            "suffix_status": suffix_generation_status,
            "suffix_forced_status": suffix_forced["status"],
            "suffix_forced_projection": suffix_forced["projection"],
            "jailbreak_confirmed": jailbreak_confirmed,
            "loss_behavior_gap": loss_behavior_gap,
        })

    return comparisons


def main():
    print("=" * 60)
    print("ACT-Break -- Step 5: Multi-Layer Suffix Optimization (GCG)")
    print("=" * 60)

    # 1. Load prompts
    prompts_data = load_prompts(str(config.ADVBENCH_PATH), max_prompts=config.OPT_NUM_PROMPTS)
    prompts = [p["goal"] for p in prompts_data]

    # 2. Load multi-layer direction vectors
    multi_path = config.MULTI_LAYER_DIRECTIONS_PATH
    single_path = config.DIRECTION_PATH

    if multi_path.exists():
        print(f"[*] Loading multi-layer directions from {multi_path}")
        multi_data = torch.load(str(multi_path), map_location="cpu")
        direction_vecs = multi_data["directions"]
        best_layer = multi_data["best_layer"]
        direction_layers = multi_data["layers"]
        print(f"[+] Loaded {len(direction_vecs)} layer directions (L{direction_layers[0]}-L{direction_layers[-1]}), best=L{best_layer}")
    elif single_path.exists():
        print(f"[!] Multi-layer directions not found. Falling back to single-layer: {single_path}")
        direction_data = torch.load(str(single_path), map_location="cpu")
        best_layer = direction_data["layer"]
        direction_vecs = {best_layer: direction_data["direction"]}
        direction_layers = [best_layer]
    else:
        print(f"[!] No direction vectors found. Run Step 3 first.")
        sys.exit(1)

    best_direction = direction_vecs[best_layer]
    activation_classifier = ActivationReferenceClassifier.from_path(
        config.ACTIVATIONS_DIR / "activations.pt",
        best_direction,
        best_layer,
    )
    print(
        "[+] Loaded Stage 1 activation reference: "
        f"L{best_layer}, refusal_mean={activation_classifier.refusal_mean:+.1f}, "
        f"compliance_mean={activation_classifier.compliance_mean:+.1f}, "
        f"threshold={activation_classifier.threshold:+.1f}"
    )

    # 3. Load model (hook ALL target layers for activation capture)
    model = HookedModel(
        model_name=config.MODEL_NAME,
        target_layers=direction_layers,
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    model.load()

    n_layers = len(direction_layers)
    print(f"\n[*] Multi-layer GCG optimization across {n_layers} layers: L{direction_layers[0]}-L{direction_layers[-1]}")
    print(f"[*] Configuration: Batch Size = {config.GCG_BATCH_SIZE}, Mini-Batch Size = {config.GCG_MINI_BATCH}, Top K = {config.GCG_TOP_K}")

    # 4. Optimize each prompt (with incremental saving)
    results = []

    for idx, prompt in enumerate(prompts):
        prompt_safe = prompt.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[#{idx+1}/{len(prompts)}] Prompt: {prompt_safe[:60]}...")

        optimizer = GCGOptimizer(
            hooked_model=model,
            direction_vecs=direction_vecs,
            direction_layers=direction_layers,
            best_layer=best_layer,
            suffix_length=config.SUFFIX_LENGTH,
            top_k=config.GCG_TOP_K,
            batch_size=config.GCG_BATCH_SIZE,
            mini_batch_size=config.GCG_MINI_BATCH,
            alpha=config.LOSS_ALPHA,
            beta=config.LOSS_BETA,
            target_string=config.TARGET_STRING,
            activation_classifier=activation_classifier
        )

        opt_res = optimizer.optimize(
            prompt=prompt,
            max_steps=config.GCG_MAX_STEPS,
            check_interval=config.GCG_CHECK_INTERVAL
        )

        result_entry = {
            "prompt": prompt,
            "forced_target_success": opt_res["success"],
            # Backward-compatible alias for older consumers.
            "success": opt_res["success"],
            "steps": opt_res["steps"],
            "suffix": opt_res["suffix"],
            "response": opt_res["response"],
            "losses": opt_res["losses"],
            "target_losses": opt_res["target_losses"],
            "activation_losses": opt_res["activation_losses"],
            "forced_target": opt_res["forced_target"],
            "generation_trajectory": opt_res["generation_trajectory"],
            "loss_behavior_gap": opt_res["loss_behavior_gap"],
        }
        results.append(result_entry)

        status_str = "FORCED-TARGET SUCCESS" if opt_res["success"] else "FAILED"
        res_safe = opt_res["response"].replace(chr(10), " ")[:70].encode("ascii", errors="replace").decode("ascii")
        print(f"[Result] {status_str} | Steps: {opt_res['steps']} | Response: {res_safe}...")

        # Incremental save after each prompt
        save_incremental(results, config.OPTIMIZATION_DIR)
        print(f"[+] Incremental save: {len(results)} prompt(s) saved to results.json")

    # 5. Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, res in enumerate(results):
        status_str = "FT-OK" if res["success"] else "FAILED"
        ax.plot(res["losses"], label=f"P{idx+1} ({status_str}, {res['steps']}s)", alpha=0.7)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(f"Multi-Layer GCG Suffix Optimization Loss Curves ({n_layers} layers)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = config.OPTIMIZATION_DIR / "loss_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved loss curves to {plot_path}")

    # 6. Run comparison test (activation-based, language-agnostic)
    comparisons = run_comparison_test(model, results, best_direction, best_layer, activation_classifier)

    # 7. Save final results with comparisons
    forced_target_success_count = sum(1 for r in results if r["success"])
    confirmed_jailbreaks = sum(1 for c in comparisons if c["jailbreak_confirmed"])
    generation_compliance_count = sum(1 for c in comparisons if c["suffix_generation"]["generated_any_compliance"])
    loss_behavior_gap_count = sum(1 for c in comparisons if c["loss_behavior_gap"])

    final_results = {
        "total_prompts": len(prompts),
        "forced_target_success_count": forced_target_success_count,
        "forced_target_success_rate": forced_target_success_count / len(prompts),
        # Backward-compatible aliases. These are objective-side successes,
        # not confirmed behavioral jailbreaks.
        "gcg_success_count": forced_target_success_count,
        "gcg_success_rate": forced_target_success_count / len(prompts),
        "confirmed_jailbreaks": confirmed_jailbreaks,
        "confirmed_jailbreak_rate": confirmed_jailbreaks / len(prompts),
        "generation_compliance_count": generation_compliance_count,
        "generation_compliance_rate": generation_compliance_count / len(prompts),
        "loss_behavior_gap_count": loss_behavior_gap_count,
        "loss_behavior_gap_rate": loss_behavior_gap_count / len(prompts),
        "multi_layer_count": n_layers,
        "direction_layers": direction_layers,
        "optimization_results": results,
        "comparison_results": comparisons
    }

    results_path = config.OPTIMIZATION_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved final results to {results_path}")

    # 8. Save text summary
    summary_path = config.OPTIMIZATION_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ACT-Break Multi-Layer GCG Optimization Summary ({n_layers} layers)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total prompts:          {len(prompts)}\n")
        f.write(f"Steering layers:        L{direction_layers[0]}-L{direction_layers[-1]} ({n_layers} layers)\n")
        f.write(f"Forced-target success:  {forced_target_success_count}/{len(prompts)} ({forced_target_success_count / len(prompts):.1%})\n")
        f.write(f"Generated compliance:   {generation_compliance_count}/{len(prompts)} ({generation_compliance_count / len(prompts):.1%})\n")
        f.write(f"Loss-behavior gaps:     {loss_behavior_gap_count}/{len(prompts)} ({loss_behavior_gap_count / len(prompts):.1%})\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Detailed Results:\n")
        f.write("-" * 60 + "\n\n")
        for idx, (res, comp) in enumerate(zip(results, comparisons)):
            f.write(f"Prompt {idx+1}: {res['prompt']}\n")
            f.write(f"  Forced-target Success: {res['success']}\n")
            f.write(f"  Steps:             {res['steps']}\n")
            f.write(f"  Suffix:            {res['suffix']}\n")
            f.write(f"  Original Gen:      {comp['original_status']}\n")
            f.write(f"  Forced Target:     {comp['suffix_forced_status']} (proj={comp['suffix_forced_projection']:+.1f})\n")
            f.write(f"  Suffix Gen:        {comp['suffix_status']}\n")
            if res.get("losses"):
                f.write(f"  Final Loss:        {res['losses'][-1]:+.4f}\n")
            if res.get("target_losses"):
                f.write(f"  Final CE Loss:     {res['target_losses'][-1]:+.4f}\n")
            if res.get("activation_losses"):
                f.write(f"  Final Act Loss:    {res['activation_losses'][-1]:+.4f}\n")
            f.write(f"  Jailbreak Confirmed: {comp['jailbreak_confirmed']}\n")
            f.write(f"  Loss-Behavior Gap: {comp['loss_behavior_gap']}\n")
            f.write(f"  Original Response: {comp['original_response'].strip().replace(chr(10), ' ')[:100]}\n")
            f.write(f"  Suffix Response:   {comp['suffix_response'].strip().replace(chr(10), ' ')[:100]}\n\n")

    print(f"[+] Saved summary text to {summary_path}")

    # 9. Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Steering Layers:        L{direction_layers[0]}-L{direction_layers[-1]} ({n_layers} layers)")
    print(f"  Forced-target Success:  {forced_target_success_count}/{len(prompts)} ({forced_target_success_count / len(prompts):.1%})")
    print(f"  Generated Compliance:   {generation_compliance_count}/{len(prompts)} ({generation_compliance_count / len(prompts):.1%})")
    print(f"  Loss-Behavior Gaps:     {loss_behavior_gap_count}/{len(prompts)} ({loss_behavior_gap_count / len(prompts):.1%})")
    print()
    print("  Comparison Summary:")
    for idx, comp in enumerate(comparisons):
        marker = "[OK]" if comp["jailbreak_confirmed"] else "[--]"
        print(
            f"    {marker} P{idx+1}: forced={comp['suffix_forced_status']} "
            f"(proj={comp['suffix_forced_projection']:+.1f}) "
            f"| gen={comp['suffix_status']} | gap={comp['loss_behavior_gap']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
