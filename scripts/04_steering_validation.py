import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.model_loader import HookedModel
from src.activation_steering import ActivationSteerer
from src.activation_collector import load_prompts
from src.activation_reference import ActivationReferenceClassifier


def classify_by_activation(model, prompt_text, direction_vec, layer_idx, activation_classifier=None):
    """
    Language-agnostic classification via activation projection.
    Forward pass on formatted prompt -> extract hidden state -> dot(act, V_jailbreak).
    Positive = compliance, negative = refusal.
    """
    formatted = model.format_chat(prompt_text, assistant_prefix=None)
    inputs = model.tokenize(formatted)

    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )

    hidden_idx = layer_idx + 1
    if outputs.hidden_states is None or hidden_idx >= len(outputs.hidden_states):
        return "Unknown", -999.0

    act_vec = outputs.hidden_states[hidden_idx][0, -1, :].float()

    proj = torch.dot(act_vec.cpu(), direction_vec.float().cpu()).item()
    if activation_classifier is not None:
        status = activation_classifier.classify_projection(proj).status
    else:
        status = "Compliance" if proj > 0 else "Refusal"
    return status, proj


def generate_without_hook_management(model, prompt_text, max_new_tokens=40):
    formatted_prompt = model.format_chat(prompt_text, assistant_prefix=None)
    inputs = model.tokenize(formatted_prompt)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    input_len = inputs["input_ids"].shape[1]
    return model.decode(outputs[0, input_len:]).strip()

def main():
    print("=" * 60)
    print("ACT-Break -- Step 4: Multi-Layer Activation Steering Validation")
    print("=" * 60)

    # 1. Load prompts
    prompts_data = load_prompts(str(config.ADVBENCH_PATH), max_prompts=config.STEERING_NUM_PROMPTS)
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
        print(f"[!] No steering vectors found. Run Step 3 first.")
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

    for idx, prompt in enumerate(prompts):
        prompt_safe = prompt.encode('ascii', errors='replace').decode('ascii')
        print(f"\n[#{idx+1}] Prompt: {prompt_safe[:60]}...")
        prompt_results = {"prompt": prompt, "sweep": {}}
        
        for alpha in alphas:
            steerer.register_hooks(alpha)
            try:
                # Activation-based classification while steering hooks are active.
                status, proj_val = classify_by_activation(
                    model, prompt, best_direction, best_layer, activation_classifier
                )
                response = generate_without_hook_management(model, prompt, max_new_tokens=40)
            finally:
                steerer.remove_hooks()
            
            snippet = response.replace('\n', ' ').strip()[:50].encode('ascii', errors='replace').decode('ascii')
            print(
                f"  alpha={alpha:>2.1f} | {status:<10} "
                f"(proj={proj_val:+.1f}) | Response: {snippet}..."
            )

            
            prompt_results["sweep"][str(alpha)] = {
                "response": response,
                "status": status,
                "projection": proj_val
            }
            
            if status == "Compliance":
                success_counts[alpha] += 1
                
        results.append(prompt_results)

    # 6. Save results
    output_file = config.STEERING_DIR / "steering_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "alphas": alphas,
            "success_counts": success_counts,
            "total_prompts": len(prompts),
            "steering_layers": layer_indices,
            "num_steering_layers": n_layers,
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Sweep Summary (Multi-Layer Steering: {n_layers} layers)")
    print("-" * 60)
    for alpha in alphas:
        count = success_counts[alpha]
        rate = count / len(prompts)
        print(f"  alpha={alpha:>4.1f} : {count:>2}/{len(prompts):>2} ({rate:>6.1%})")
    print("=" * 60)
    print(f"[+] Saved results to {output_file}")

if __name__ == "__main__":
    main()
