import csv
import torch
from tqdm import tqdm

def load_prompts(csv_path: str, max_prompts: int | None = None) -> list[dict]:
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({"goal": row["goal"], "target": row["target"]})

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    print(f"[+] Loaded {len(prompts)} prompts from {csv_path}")
    return prompts

def extract_last_token_activation(store, target_layers: list[int]) -> dict[int, torch.Tensor]:
    result = {}
    for layer_idx in target_layers:
        act = store.get(layer_idx)
        if act is None:
            raise ValueError(f"No activation found for layer {layer_idx}")
        # Extract at the last token position: [hidden_dim]
        if act.dim() == 3:
            result[layer_idx] = act[0, -1, :].clone()
        elif act.dim() == 2:
            result[layer_idx] = act[-1, :].clone()
        else:
            raise ValueError(f"Unexpected activation shape for layer {layer_idx}: {tuple(act.shape)}")

    return result

def collect_activations(hooked_model, prompts: list[dict], target_layers: list[int],
                        compliance_prefix: str, max_new_tokens: int = 20):
    all_activations = {layer: [] for layer in target_layers}
    all_labels = []
    refusal_responses = []

    print("\n" + "="*50)
    print("Collecting Contrastive Activations")
    print("="*50 + "\n")

    for i, prompt_data in enumerate(tqdm(prompts, desc="Collecting activations")):
        goal = prompt_data["goal"]

        # 1. Refusal Pass (Normal Generation)
        hooked_model.store.clear()
        refusal_text = hooked_model.format_chat(goal, assistant_prefix=None)
        refusal_inputs = hooked_model.tokenize(refusal_text)

        output_ids = hooked_model.generate(
            refusal_inputs["input_ids"],
            attention_mask=refusal_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
        )
        generated_text = hooked_model.decode(output_ids[0][refusal_inputs["input_ids"].shape[1]:])
        refusal_responses.append(generated_text)

        # Capture refusal activations from forward pass
        hooked_model.store.clear()
        hooked_model.forward(
            refusal_inputs["input_ids"],
            attention_mask=refusal_inputs.get("attention_mask"),
        )
        refusal_acts = extract_last_token_activation(hooked_model.store, target_layers)

        for layer in target_layers:
            all_activations[layer].append(refusal_acts[layer])
        all_labels.append(0)

        # 2. Compliance Pass (Forced Response Prefix)
        hooked_model.store.clear()
        compliance_text = hooked_model.format_chat(goal, assistant_prefix=compliance_prefix)
        compliance_inputs = hooked_model.tokenize(compliance_text)

        hooked_model.forward(
            compliance_inputs["input_ids"],
            attention_mask=compliance_inputs.get("attention_mask"),
        )
        compliance_acts = extract_last_token_activation(hooked_model.store, target_layers)

        for layer in target_layers:
            all_activations[layer].append(compliance_acts[layer])
        all_labels.append(1)

        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"  Prompt: {goal[:60]}...")
            print(f"  Refusal Response: {generated_text.strip()[:60]}...")
            print(f"  Compliance Prefix: '{compliance_prefix}'")

    result = {
        "activations": {},
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "prompts": [p["goal"] for p in prompts],
        "refusal_responses": refusal_responses,
    }

    for layer in target_layers:
        result["activations"][layer] = torch.stack(all_activations[layer])

    return result

def save_activations(result: dict, output_path: str):
    torch.save(result, output_path)
    import os
    print(f"[+] Saved activations to: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.1f} MB)")

def load_activations(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    n_samples = len(data["labels"])
    n_layers = len(data["activations"])
    print(f"[+] Loaded activations from: {path} ({n_samples} samples, {n_layers} layers)")
    return data
