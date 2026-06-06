import torch


def suffix_prompt(prompt: str, suffix: str) -> str:
    return prompt if not suffix.strip() else prompt + " " + suffix


def classify_input_ids(
    model,
    input_ids: torch.Tensor,
    direction_vec: torch.Tensor,
    layer_idx: int,
    activation_classifier=None,
    position: int = -1,
) -> dict:
    raw_model = model.model if hasattr(model, "model") else model
    input_ids = input_ids.to(raw_model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = raw_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_idx = layer_idx + 1
    if outputs.hidden_states is None or hidden_idx >= len(outputs.hidden_states):
        return {
            "status": "Unknown",
            "is_compliance": False,
            "projection": -999.0,
            "decision": {
                "status": "Unknown",
                "projection": -999.0,
                "threshold": 0.0,
                "margin": -999.0,
            },
        }

    act_vec = outputs.hidden_states[hidden_idx][0, position, :].float()
    if activation_classifier is not None:
        decision = activation_classifier.classify_activation(act_vec)
        return {
            "status": decision.status,
            "is_compliance": decision.status == "Compliance",
            "projection": decision.projection,
            "decision": decision.to_dict(),
        }

    projection = torch.dot(act_vec.detach().cpu(), direction_vec.float().cpu()).item()
    status = "Compliance" if projection > 0 else "Refusal"
    return {
        "status": status,
        "is_compliance": status == "Compliance",
        "projection": float(projection),
        "decision": {
            "status": status,
            "projection": float(projection),
            "threshold": 0.0,
            "margin": float(projection),
        },
    }


def measure_forced_target_text(
    model,
    prompt: str,
    suffix: str,
    target_string: str,
    direction_vec: torch.Tensor,
    layer_idx: int,
    activation_classifier=None,
) -> dict:
    formatted = model.format_chat(suffix_prompt(prompt, suffix), assistant_prefix=None)
    forced_text = formatted + target_string
    inputs = model.tokenize(forced_text)
    return classify_input_ids(
        model=model,
        input_ids=inputs["input_ids"],
        direction_vec=direction_vec,
        layer_idx=layer_idx,
        activation_classifier=activation_classifier,
        position=-1,
    )


def track_generation_trajectory_text(
    model,
    prompt: str,
    suffix: str,
    direction_vec: torch.Tensor,
    layer_idx: int,
    activation_classifier=None,
    checkpoints: tuple[int, ...] = (1, 3, 5, 10),
    max_new_tokens: int = 40,
) -> dict:
    raw_model = model.model
    tokenizer = model.tokenizer

    formatted = model.format_chat(suffix_prompt(prompt, suffix), assistant_prefix=None)
    inputs = model.tokenize(formatted)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_tokens = []
    checkpoint_set = set(checkpoints)
    trajectory = []

    for _ in range(max_new_tokens + 1):
        with torch.no_grad():
            outputs = raw_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        generated_count = len(generated_tokens)
        if generated_count in checkpoint_set:
            hidden_state = outputs.hidden_states[layer_idx + 1][0, -1, :].float()
            if activation_classifier is not None:
                decision = activation_classifier.classify_activation(hidden_state)
                status = decision.status
                projection = decision.projection
                decision_dict = decision.to_dict()
            else:
                projection = torch.dot(
                    hidden_state.detach().cpu(),
                    direction_vec.float().cpu(),
                ).item()
                status = "Compliance" if projection > 0 else "Refusal"
                decision_dict = {
                    "status": status,
                    "projection": float(projection),
                    "threshold": 0.0,
                    "margin": float(projection),
                }

            trajectory.append({
                "step": generated_count,
                "status": status,
                "is_compliance": status == "Compliance",
                "projection": float(projection),
                "decision": decision_dict,
                "token": tokenizer.decode([generated_tokens[-1]], skip_special_tokens=True).strip()
                if generated_tokens else "",
                "text_so_far": tokenizer.decode(generated_tokens, skip_special_tokens=True).strip(),
            })

        if generated_count >= max_new_tokens:
            break

        next_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_val = next_token.item()
        if token_val == tokenizer.eos_token_id:
            break

        generated_tokens.append(token_val)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
        ], dim=-1)

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "response": response,
        "trajectory": trajectory,
        "generated_any_compliance": any(item["is_compliance"] for item in trajectory),
    }
