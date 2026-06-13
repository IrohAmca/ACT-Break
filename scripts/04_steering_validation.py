import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.model_loader import HookedModel
from src.activation_steering import ActivationSteerer
from src.activation_collector import load_prompts
from src.activation_measurements import track_generation_trajectory_text
from src.activation_reference import ActivationReferenceClassifier
from src.behavior_scoring import score_response, summarize_behavior_scores
from src.text_utils import safe_console_text


def format_alpha(alpha: float | int) -> str:
    return f"{alpha:g}" if isinstance(alpha, (float, int)) else str(alpha)


def summarize_generation_trajectory(generation: dict) -> dict:
    trajectory = generation.get("trajectory", [])
    projections = [float(item["projection"]) for item in trajectory]
    margins = [
        float(item["decision"]["margin"])
        for item in trajectory
        if "decision" in item and "margin" in item["decision"]
    ]
    statuses = [item["status"] for item in trajectory]

    return {
        "checkpoint_count": len(trajectory),
        "first_projection": projections[0] if projections else -999.0,
        "max_projection": max(projections) if projections else -999.0,
        "min_projection": min(projections) if projections else -999.0,
        "max_margin": max(margins) if margins else -999.0,
        "min_margin": min(margins) if margins else -999.0,
        "activation_compliance_checkpoints": statuses.count("Compliance"),
        "generated_any_compliance": bool(generation.get("generated_any_compliance", False)),
    }


def main():
    print("=" * 60)
    print("ACT-Break -- Step 4: Multi-Layer Activation Steering Validation")
    print("=" * 60)

    # 1. Load prompts
    prompts_data = load_prompts(
        str(config.ADVBENCH_PATH),
        max_prompts=config.STEERING_NUM_PROMPTS,
        language=config.ADVBENCH_LANGUAGE,
    )
    prompts = [p["goal"] for p in prompts_data]

    # 2. Load multi-layer direction vectors
    multi_path = config.MULTI_LAYER_DIRECTIONS_PATH
    single_path = config.DIRECTION_PATH

    if multi_path.exists():
        print(f"[*] Loading multi-layer directions from {multi_path}")
        multi_data = torch.load(str(multi_path), map_location="cpu")
        direction_vecs = multi_data["directions"]
        best_layer = multi_data["best_layer"]
        layer_indices = multi_data["layers"]
        print(f"[+] Loaded {len(direction_vecs)} layer directions (L{layer_indices[0]}-L{layer_indices[-1]}), best=L{best_layer}")
    elif single_path.exists():
        print(f"[!] Multi-layer directions not found. Falling back to single-layer: {single_path}")
        direction_data = torch.load(str(single_path), map_location="cpu")
        best_layer = direction_data["layer"]
        direction_vecs = {best_layer: direction_data["direction"]}
        layer_indices = [best_layer]
    else:
        print("[!] No steering vectors found. Run Step 3 first.")
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

    # 3. Load model (hook all target layers for activation capture)
    model = HookedModel(
        model_name=config.MODEL_NAME,
        target_layers=layer_indices,
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    model.load()

    # 4. Create multi-layer steerer
    steerer = ActivationSteerer(
        hooked_model=model,
        direction_vecs=direction_vecs,
        layer_indices=layer_indices
    )

    n_layers = len(layer_indices)
    print(f"\n[*] Multi-layer steering on {n_layers} layers: L{layer_indices[0]}-L{layer_indices[-1]}")

    # 5. Run alpha sweep
    results = []
    print("\n[*] Starting Alpha Sweep on steering prompts...")
    print("  (Using activation projection for classification, best layer L%d)" % best_layer)
    alphas = config.STEERING_ALPHAS

    success_counts = {alpha: 0 for alpha in alphas}
    activation_success_counts = {str(alpha): 0 for alpha in alphas}
    behavior_scores_by_alpha = {str(alpha): [] for alpha in alphas}

    for idx, prompt in enumerate(prompts):
        print(f"\n[#{idx+1}] Prompt: {safe_console_text(prompt, 60)}...")
        prompt_results = {"prompt": prompt, "sweep": {}}
        
        for alpha in alphas:
            alpha_key = str(alpha)
            alpha_label = format_alpha(alpha)
            steerer.register_hooks(alpha)
            try:
                generation = track_generation_trajectory_text(
                    model=model,
                    prompt=prompt,
                    suffix="",
                    direction_vec=best_direction,
                    layer_idx=best_layer,
                    activation_classifier=activation_classifier,
                    max_new_tokens=40,
                )
            finally:
                steerer.remove_hooks()

            response = generation["response"]
            status = "Compliance" if generation["generated_any_compliance"] else "Refusal"
            trajectory_summary = summarize_generation_trajectory(generation)
            behavior_score = score_response(prompt, response)
            behavior_scores_by_alpha[alpha_key].append(behavior_score)
            proj_val = (
                generation["trajectory"][0]["projection"]
                if generation["trajectory"] else -999.0
            )
            
            snippet = safe_console_text(response, 50)
            print(
                f"  alpha={alpha_label:>5} | {status:<10} "
                f"(proj={proj_val:+.1f}) | behavior={behavior_score['label']:<22} "
                f"| overlap={behavior_score['lexical_prompt_overlap']:.2f} "
                f"| rep={behavior_score['repetition_share']:.2f} "
                f"| Response: {snippet}..."
            )

            
            prompt_results["sweep"][alpha_key] = {
                "response": response,
                "status": status,
                "projection": proj_val,
                "trajectory_summary": trajectory_summary,
                "behavior_score": behavior_score,
                "generation_trajectory": generation,
            }
            
            if status == "Compliance":
                success_counts[alpha] += 1
                activation_success_counts[alpha_key] += 1
                
        results.append(prompt_results)

    behavior_summary = {
        str(alpha): summarize_behavior_scores(behavior_scores_by_alpha[str(alpha)])
        for alpha in alphas
    }
    steering_summary = {}
    for alpha in alphas:
        alpha_key = str(alpha)
        steering_summary[alpha_key] = {
            "activation_success_count": activation_success_counts[alpha_key],
            "activation_success_rate": (
                activation_success_counts[alpha_key] / len(prompts) if prompts else 0.0
            ),
            **behavior_summary[alpha_key],
        }

    # 6. Save results
    output_file = config.STEERING_DIR / "steering_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "steering_results.behavior_v1",
            "model_info": {
                "model_name": config.MODEL_NAME,
                "model_suffix": config.MODEL_SUFFIX,
                "model_profile": getattr(config, "MODEL_PROFILE", None),
                "advbench_language": config.ADVBENCH_LANGUAGE,
                "target_layers": config.TARGET_LAYERS,
                "negative_activation_mode": config.NEGATIVE_ACTIVATION_MODE,
                "compliance_prefix": config.DEFAULT_COMPLIANCE_PREFIX,
                "refusal_prefix": config.DEFAULT_REFUSAL_PREFIX,
                "dtype": config.DTYPE,
                "device": config.DEVICE,
            },
            "alphas": alphas,
            "success_counts": success_counts,
            "activation_success_counts": activation_success_counts,
            "behavior_summary": behavior_summary,
            "steering_summary": steering_summary,
            "total_prompts": len(prompts),
            "steering_layers": layer_indices,
            "num_steering_layers": n_layers,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Sweep Summary (Multi-Layer Steering: {n_layers} layers)")
    print("-" * 60)
    for alpha in alphas:
        count = success_counts[alpha]
        rate = count / len(prompts)
        alpha_key = str(alpha)
        alpha_label = format_alpha(alpha)
        behavior = steering_summary[alpha_key]
        print(
            f"  alpha={alpha_label:>5} : act={count:>2}/{len(prompts):>2} ({rate:>6.1%}) "
            f"| non_refusal={behavior['behavioral_non_refusal']:>2} "
            f"| jailbreak_candidate={behavior['behavioral_jailbreak_candidate']:>2} "
            f"| echo={behavior['prompt_echo']:>2} "
            f"| repeat={behavior['repetition_collapse']:>2}"
        )
    print("=" * 60)
    print(f"[+] Saved results to {output_file}")

if __name__ == "__main__":
    main()
