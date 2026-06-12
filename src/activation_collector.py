from __future__ import annotations

import csv
from tqdm import tqdm

def load_prompts(csv_path: str, max_prompts: int | None = None, language: str | None = None) -> list[dict]:
    language = (language or "en").lower()
    goal_column = "goal_tr" if language == "tr" else "goal"
    target_column = "target_tr" if language == "tr" else "target"

    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if goal_column not in fieldnames or target_column not in fieldnames:
            if language == "tr" and {"goal", "target"}.issubset(fieldnames):
                goal_column = "goal"
                target_column = "target"
            else:
                raise ValueError(
                    f"{csv_path} must contain `{goal_column}` and `{target_column}` columns "
                    f"for language={language!r}. Found columns: {sorted(fieldnames)}"
                )

        for row in reader:
            prompts.append({"goal": row[goal_column], "target": row[target_column]})

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    print(f"[+] Loaded {len(prompts)} prompts from {csv_path} ({language}, {goal_column}/{target_column})")
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

def first_n_decoded_tokens(tokenizer, text: str, token_count: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text.strip()
    return tokenizer.decode(token_ids[:token_count], skip_special_tokens=True).strip()

def collect_activations(hooked_model, prompts: list[dict], target_layers: list[int],
                        compliance_prefix: str, max_new_tokens: int = 20):
    import torch

    all_activations = {layer: [] for layer in target_layers}
    all_labels = []
    refusal_responses = []
    refusal_prefixes = []
    compliance_token_count = len(
        hooked_model.tokenizer.encode(compliance_prefix, add_special_tokens=False)
    )

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

        # Capture refusal/compliance activations at the same response depth.
        refusal_prefix = first_n_decoded_tokens(
            hooked_model.tokenizer,
            generated_text,
            max(1, compliance_token_count),
        )
        refusal_prefixes.append(refusal_prefix)
        assistant_prompt_text = hooked_model.format_chat(goal, assistant_prefix=None)
        refusal_prefix_text = assistant_prompt_text + refusal_prefix
        refusal_prefix_inputs = hooked_model.tokenize(refusal_prefix_text)

        hooked_model.store.clear()
        hooked_model.forward(
            refusal_prefix_inputs["input_ids"],
            attention_mask=refusal_prefix_inputs.get("attention_mask"),
        )
        refusal_acts = extract_last_token_activation(hooked_model.store, target_layers)

        for layer in target_layers:
            all_activations[layer].append(refusal_acts[layer])
        all_labels.append(0)

        # 2. Compliance Pass (Forced Response Prefix)
        hooked_model.store.clear()
        compliance_text = assistant_prompt_text + compliance_prefix
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
            print(f"  Refusal Prefix: '{refusal_prefix}'")
            print(f"  Compliance Prefix: '{compliance_prefix}'")

    result = {
        "activations": {},
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "prompts": [p["goal"] for p in prompts],
        "refusal_responses": refusal_responses,
        "refusal_prefixes": refusal_prefixes,
        "compliance_prefix": compliance_prefix,
        "activation_position": "assistant_prefix_last_token",
    }

    for layer in target_layers:
        result["activations"][layer] = torch.stack(all_activations[layer])

    return result

def save_activations(result: dict, output_path: str):
    torch.save(result, output_path)
    import os
    print(f"[+] Saved activations to: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.1f} MB)")

def load_activations(path: str) -> dict:
    import torch

    data = torch.load(path, weights_only=False)
    n_samples = len(data["labels"])
    n_layers = len(data["activations"])
    print(f"[+] Loaded activations from: {path} ({n_samples} samples, {n_layers} layers)")
    return data
