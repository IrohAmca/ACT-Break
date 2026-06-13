import random
import math
import torch
from src.behavior_scoring import score_response
from src.token_gradients import compute_token_gradients
from src.loss_functions import compute_loss
from src.text_utils import safe_console_text


def normalize_target_strings(target_string: str, target_strings: list[str] | None = None) -> list[str]:
    targets = list(target_strings or [])
    if target_string:
        targets.insert(0, target_string)

    deduped = []
    seen = set()
    for target in targets:
        target = target.strip()
        if target and target not in seen:
            deduped.append(target)
            seen.add(target)

    if not deduped:
        raise ValueError("At least one target string is required for GCG optimization.")
    return deduped


def behavior_gate_passed(behavior_score: dict) -> bool:
    return bool(
        behavior_score.get("behavioral_jailbreak_candidate")
        and not behavior_score.get("prompt_echo")
        and not behavior_score.get("repetition_collapse")
    )


def prune_target_length(suffix_length: int, prune_fraction: float, min_tokens: int) -> int:
    if suffix_length <= 1:
        return max(1, suffix_length)
    prune_fraction = min(max(float(prune_fraction), 0.0), 0.95)
    min_tokens = max(1, int(min_tokens))
    target = math.ceil(suffix_length * (1.0 - prune_fraction))
    return max(1, min(suffix_length, max(min_tokens, target)))


class GCGOptimizer:
    def __init__(self, hooked_model, direction_vecs, direction_layers, best_layer,
                 suffix_length=20, top_k=256, batch_size=64, mini_batch_size=16,
                 alpha=1.0, beta=0.3, target_string="Sure, here is",
                 activation_classifier=None, target_strings: list[str] | None = None,
                 max_coordinate_updates: int = 1, deduplicate_candidates: bool = True,
                 require_behavior_for_early_stop: bool = False,
                 archive_top_n: int = 0, archive_limit: int = 0,
                 token_distance_weight: float = 0.0, momentum: float = 0.0,
                 enable_prune_refine: bool = False, prune_fraction: float = 0.0,
                 prune_min_tokens: int = 1, prune_refine_steps: int = 0,
                 prune_max_rel_loss_increase: float = 0.03):
        """
        Multi-layer GCG optimizer.

        Args:
            hooked_model: HookedModel instance
            direction_vecs: dict mapping layer_idx -> direction Tensor [hidden_dim]
            direction_layers: list of int, e.g. [8, 9, 10, 11, 12, 13, 14]
            best_layer: int, the best layer index for classify_by_activation
            suffix_length, top_k, batch_size, mini_batch_size, alpha, beta: GCG hyperparams
        """
        self.model = hooked_model
        self.tokenizer = hooked_model.tokenizer
        self.raw_model = hooked_model.model
        self.direction_vecs = direction_vecs
        self.direction_layers = direction_layers
        self.best_layer = best_layer
        
        self.suffix_length = suffix_length
        self.top_k = top_k
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.alpha = alpha
        self.beta = beta
        self.target_strings = normalize_target_strings(target_string, target_strings)
        self.target_string = self.target_strings[0]
        self.activation_classifier = activation_classifier
        self.max_coordinate_updates = max(1, min(int(max_coordinate_updates), suffix_length))
        self.deduplicate_candidates = deduplicate_candidates
        self.require_behavior_for_early_stop = require_behavior_for_early_stop
        self.archive_top_n = max(0, int(archive_top_n))
        self.archive_limit = max(0, int(archive_limit))
        self.token_distance_weight = max(0.0, float(token_distance_weight))
        self.momentum = min(max(float(momentum), 0.0), 0.999)
        self.enable_prune_refine = enable_prune_refine
        self.prune_fraction = min(max(float(prune_fraction), 0.0), 0.95)
        self.prune_min_tokens = max(1, int(prune_min_tokens))
        self.prune_refine_steps = max(0, int(prune_refine_steps))
        self.prune_max_rel_loss_increase = max(0.0, float(prune_max_rel_loss_increase))
        self._momentum_grad = None
        self.candidate_archive = []
        self._archive_suffixes = set()
        self._seen_suffixes = set()
        
        # Initialize suffix with '!' token repeated
        init_token_ids = self.tokenizer.encode("!", add_special_tokens=False)
        if not init_token_ids:
            init_token_ids = [self.tokenizer.eos_token_id]
        init_id = init_token_ids[0]
        
        self.suffix_ids = torch.tensor([init_id] * self.suffix_length, dtype=torch.long, device=self.raw_model.device)
        self._seen_suffixes.add(tuple(self.suffix_ids.tolist()))
        
    def decode_suffix(self) -> str:
        return self._decode_ids(self.suffix_ids)

    def _decode_ids(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _split_formatted_user_slot(self, prompt: str) -> tuple[str, str]:
        sentinel = "ACT_BREAK_SUFFIX_SLOT"
        if sentinel in prompt:
            raise ValueError("Prompt contains reserved ACT-Break suffix sentinel.")

        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + " " + sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if sentinel not in formatted:
            raise ValueError("Could not locate suffix sentinel in chat template output.")

        prefix_text, post_suffix_text = formatted.split(sentinel, 1)
        return prefix_text, post_suffix_text
        
    def _select_target_string(self, step_idx: int | None = None) -> str:
        if step_idx is None:
            return self.target_string
        return self.target_strings[(step_idx - 1) % len(self.target_strings)]

    def build_input(
        self,
        prompt: str,
        target_string: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes the prompt into prefix/suffix/post-suffix/target segments.

        The suffix slot is inside the user message, matching the final
        suffix-injected evaluation prompt.
        """
        prefix_text, post_suffix_text = self._split_formatted_user_slot(prompt)

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        post_suffix_ids = self.tokenizer.encode(post_suffix_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_string or self.target_string, add_special_tokens=False)
        
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=self.raw_model.device)
        post_suffix_tensor = torch.tensor(post_suffix_ids, dtype=torch.long, device=self.raw_model.device)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=self.raw_model.device)
        
        return prefix_tensor, self.suffix_ids, post_suffix_tensor, target_tensor

    def build_generation_input_ids(self, prompt: str, suffix_ids: torch.Tensor | None = None) -> torch.Tensor:
        prefix_ids, current_suffix_ids, post_suffix_ids, _ = self.build_input(prompt)
        if suffix_ids is None:
            suffix_ids = current_suffix_ids
        return torch.cat([prefix_ids, suffix_ids.to(prefix_ids.device), post_suffix_ids], dim=0).unsqueeze(0)

    def build_forced_target_input_ids(
        self,
        prompt: str,
        suffix_ids: torch.Tensor | None = None,
        target_string: str | None = None,
    ) -> torch.Tensor:
        prefix_ids, current_suffix_ids, post_suffix_ids, target_ids = self.build_input(
            prompt,
            target_string=target_string,
        )
        if suffix_ids is None:
            suffix_ids = current_suffix_ids
        return torch.cat([
            prefix_ids,
            suffix_ids.to(prefix_ids.device),
            post_suffix_ids,
            target_ids,
        ], dim=0).unsqueeze(0)

    def _apply_token_distance_regularization(
        self,
        grad: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_distance_weight <= 0:
            return grad

        with torch.no_grad():
            embedding_matrix = self.raw_model.model.embed_tokens.weight.detach().float()
            current_embeds = embedding_matrix[suffix_ids.to(embedding_matrix.device)]
            vocab_norms = embedding_matrix.pow(2).sum(dim=1)
            current_norms = current_embeds.pow(2).sum(dim=1, keepdim=True)
            distances = current_norms + vocab_norms.unsqueeze(0) - 2 * current_embeds @ embedding_matrix.T
            distances = distances.clamp_min(0)
            scale = distances.mean(dim=1, keepdim=True).clamp_min(1e-6)
            distances = distances / scale

        return grad + self.token_distance_weight * distances.to(grad.device, dtype=grad.dtype)

    def _apply_momentum(self, grad: torch.Tensor, suffix_ids: torch.Tensor) -> torch.Tensor:
        if self.momentum <= 0:
            return grad

        finite_grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = finite_grad.norm(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_grad = finite_grad / grad_norm

        if (
            self._momentum_grad is None
            or self._momentum_grad.shape != normalized_grad.shape
        ):
            self._momentum_grad = normalized_grad
        else:
            self._momentum_grad = self.momentum * self._momentum_grad + normalized_grad

        return self._momentum_grad.to(grad.device, dtype=grad.dtype)

    def _position_order(self, grad: torch.Tensor, suffix_ids: torch.Tensor) -> list[int]:
        current_token_grads = grad[
            torch.arange(len(suffix_ids), device=grad.device),
            suffix_ids.to(grad.device),
        ]
        eligible = torch.nonzero(current_token_grads > 0, as_tuple=False).flatten()
        if eligible.numel() == 0:
            eligible = torch.arange(len(suffix_ids), device=grad.device)

        ordered = eligible[
            torch.argsort(current_token_grads[eligible], descending=True)
        ]
        return [int(pos) for pos in ordered.detach().cpu().tolist()]

    def _build_candidates(
        self,
        suffix_ids: torch.Tensor,
        top_k_indices: torch.Tensor,
        position_order: list[int],
    ) -> list[torch.Tensor]:
        candidates = []
        step_seen = set()
        current_key = tuple(suffix_ids.tolist())
        max_updates = min(self.max_coordinate_updates, max(1, len(position_order)))

        def add_candidate(candidate: torch.Tensor, allow_current: bool = False) -> bool:
            key = tuple(candidate.tolist())
            if key == current_key and not allow_current:
                return False
            if self.deduplicate_candidates and key in step_seen:
                return False
            if self.deduplicate_candidates and key in self._seen_suffixes and key != current_key:
                return False
            step_seen.add(key)
            candidates.append(candidate)
            return True

        attempts = 0
        max_attempts = max(self.batch_size * 20, len(position_order) * max_updates * 2)
        while len(candidates) < self.batch_size and attempts < max_attempts:
            update_count = 1 + (attempts % max_updates)
            start = (attempts // max_updates) % max(1, len(position_order))
            token_rank_base = (
                attempts // max(1, max_updates * len(position_order))
            ) % top_k_indices.shape[1]

            candidate = suffix_ids.clone()
            for offset in range(update_count):
                pos = position_order[(start + offset) % len(position_order)]
                token_rank = (token_rank_base + offset) % top_k_indices.shape[1]
                candidate[pos] = top_k_indices[pos, token_rank]

            add_candidate(candidate)
            attempts += 1

        random_attempts = 0
        max_random_attempts = self.batch_size * 40
        while len(candidates) < self.batch_size and random_attempts < max_random_attempts:
            update_count = random.randint(1, max_updates)
            positions = random.sample(position_order, k=min(update_count, len(position_order)))
            candidate = suffix_ids.clone()
            for pos in positions:
                candidate[pos] = random.choice(top_k_indices[pos].tolist())
            add_candidate(candidate)
            random_attempts += 1

        add_candidate(suffix_ids.clone(), allow_current=True)
        return candidates

    def _archive_candidates(
        self,
        prompt: str,
        target_string: str,
        candidates: list[torch.Tensor],
        losses: list[float],
        target_losses: list[float],
        activation_losses: list[float],
    ) -> None:
        if self.archive_top_n <= 0 or self.archive_limit <= 0 or not candidates:
            return

        ranked = sorted(range(len(candidates)), key=lambda idx: losses[idx])
        for idx in ranked[: self.archive_top_n]:
            suffix_text = self._decode_ids(candidates[idx])
            key = (target_string, suffix_text)
            if key in self._archive_suffixes:
                continue
            self._archive_suffixes.add(key)
            self.candidate_archive.append({
                "prompt": prompt,
                "target_string": target_string,
                "loss": float(losses[idx]),
                "target_loss": float(target_losses[idx]),
                "activation_loss": float(activation_losses[idx]),
                "suffix": suffix_text,
            })

        self.candidate_archive.sort(key=lambda item: item["loss"])
        if len(self.candidate_archive) > self.archive_limit:
            self.candidate_archive = self.candidate_archive[: self.archive_limit]
            self._archive_suffixes = {
                (item["target_string"], item["suffix"])
                for item in self.candidate_archive
            }

    def _evaluate_suffix_candidates(
        self,
        prompt: str,
        target_string: str,
        candidates: list[torch.Tensor],
    ) -> tuple[list[float], list[float], list[float]]:
        if not candidates:
            return [], [], []

        prefix_ids, _, post_suffix_ids, target_ids = self.build_input(
            prompt,
            target_string=target_string,
        )
        suffix_len = len(candidates[0])
        if any(len(candidate) != suffix_len for candidate in candidates):
            raise ValueError("All suffix candidates in a batch must have the same token length.")

        embed_tokens = self.raw_model.model.embed_tokens
        prefix_len = len(prefix_ids)
        post_suffix_len = len(post_suffix_ids)
        target_len = len(target_ids)

        suffix_slice = slice(prefix_len, prefix_len + suffix_len)
        target_start = prefix_len + suffix_len + post_suffix_len
        target_slice = slice(target_start, target_start + target_len)

        candidate_losses = []
        candidate_target_losses = []
        candidate_activation_losses = []

        for i in range(0, len(candidates), self.mini_batch_size):
            mini_batch = candidates[i:i + self.mini_batch_size]

            batch_ids = torch.stack([
                torch.cat([prefix_ids, cand.to(prefix_ids.device), post_suffix_ids, target_ids])
                for cand in mini_batch
            ])

            with torch.no_grad():
                batch_embeds = embed_tokens(batch_ids)
                outputs = self.raw_model(inputs_embeds=batch_embeds, output_hidden_states=True)

                losses, target_losses, activation_losses = compute_loss(
                    logits=outputs.logits,
                    hidden_states=outputs.hidden_states,
                    input_ids=batch_ids,
                    suffix_slice=suffix_slice,
                    target_slice=target_slice,
                    direction_vecs=self.direction_vecs,
                    direction_layers=self.direction_layers,
                    alpha=self.alpha,
                    beta=self.beta,
                    aggregation='mean'
                )
                candidate_losses.extend(losses.tolist())
                candidate_target_losses.extend(target_losses.tolist())
                candidate_activation_losses.extend(activation_losses.tolist())

            torch.cuda.empty_cache()

        return candidate_losses, candidate_target_losses, candidate_activation_losses

    def step(self, prompt: str, step_idx: int = 1) -> dict:
        """
        Executes a single GCG iteration with multi-layer gradient computation.
        """
        target_string = self._select_target_string(step_idx)
        prefix_ids, suffix_ids, post_suffix_ids, target_ids = self.build_input(
            prompt,
            target_string=target_string,
        )
        
        # 1. Compute gradients (now using all target layers)
        grad = compute_token_gradients(
            model=self.model,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            target_ids=target_ids,
            direction_vecs=self.direction_vecs,
            direction_layers=self.direction_layers,
            alpha=self.alpha,
            beta=self.beta,
            post_suffix_ids=post_suffix_ids
        )
        grad = self._apply_momentum(grad, suffix_ids)
        
        # 2. Exclude special tokens from candidates
        for special_id in self.tokenizer.all_special_ids:
            if special_id < grad.shape[1]:
                grad[:, special_id] = float("inf")

        candidate_grad = self._apply_token_distance_regularization(grad, suffix_ids)

        # 3. Find promising replacement tokens and update positions.
        top_k_indices = (-candidate_grad).topk(self.top_k, dim=1).indices # [suffix_len, top_k]
        position_order = self._position_order(candidate_grad, suffix_ids)

        # 4. Build mostly deterministic candidates, with random fallback for diversity.
        candidates = self._build_candidates(suffix_ids, top_k_indices, position_order)

        # 5. Evaluate candidates in VRAM-efficient mini-batches.
        candidate_losses, candidate_target_losses, candidate_activation_losses = (
            self._evaluate_suffix_candidates(prompt, target_string, candidates)
        )

        self._archive_candidates(
            prompt=prompt,
            target_string=target_string,
            candidates=candidates,
            losses=candidate_losses,
            target_losses=candidate_target_losses,
            activation_losses=candidate_activation_losses,
        )
            
        # 6. Select the best candidate suffix
        best_idx = torch.tensor(candidate_losses).argmin().item()
        self.suffix_ids = candidates[best_idx]
        self._seen_suffixes.add(tuple(self.suffix_ids.tolist()))
        best_loss = candidate_losses[best_idx]
        
        return {
            "loss": best_loss,
            "target_loss": candidate_target_losses[best_idx],
            "activation_loss": candidate_activation_losses[best_idx],
            "suffix": self.decode_suffix(),
            "target_string": target_string,
            "candidate_count": len(candidates),
            "updated_positions": position_order[: self.max_coordinate_updates],
        }

    def classify_activation_vector(self, act_vec: torch.Tensor) -> tuple[bool, float, str, dict]:
        if self.activation_classifier is not None:
            decision = self.activation_classifier.classify_activation(act_vec)
            return (
                decision.status == "Compliance",
                decision.projection,
                decision.status,
                decision.to_dict(),
            )

        best_dir = self.direction_vecs[self.best_layer]
        proj = torch.dot(act_vec.detach().float().cpu(), best_dir.float().cpu()).item()
        status = "Compliance" if proj > 0 else "Refusal"
        return status == "Compliance", proj, status, {
            "status": status,
            "projection": float(proj),
            "threshold": 0.0,
            "margin": float(proj),
        }

    def classify_by_activation_ids(self, input_ids: torch.Tensor, position: int = -1) -> tuple[bool, float, str, dict]:
        """
        Language-agnostic classification using activation projection.
        
        Uses the best_layer direction vector for classification.
        Positive projection = compliance, negative = refusal.
        """
        input_ids = input_ids.to(self.raw_model.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.raw_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_idx = self.best_layer + 1
        if outputs.hidden_states is None or hidden_idx >= len(outputs.hidden_states):
            return False, -999.0, "Unknown", {
                "status": "Unknown",
                "projection": -999.0,
                "threshold": 0.0,
                "margin": -999.0,
            }

        act_vec = outputs.hidden_states[hidden_idx][0, position, :].float()
        return self.classify_activation_vector(act_vec)

    def classify_by_activation(self, prompt_text: str) -> tuple[bool, float, str, dict]:
        formatted = self.model.format_chat(prompt_text, assistant_prefix=None)
        inputs = self.model.tokenize(formatted)
        return self.classify_by_activation_ids(inputs["input_ids"])

    def measure_forced_target(self, prompt: str, target_string: str | None = None) -> dict:
        target_string = target_string or self.target_string
        input_ids = self.build_forced_target_input_ids(prompt, target_string=target_string)
        is_compliance, proj_val, status, decision = self.classify_by_activation_ids(input_ids, position=-1)
        return {
            "status": status,
            "is_compliance": is_compliance,
            "projection": proj_val,
            "decision": decision,
            "target_string": target_string,
        }

    def measure_forced_targets(self, prompt: str) -> dict:
        measurements = [
            self.measure_forced_target(prompt, target_string=target_string)
            for target_string in self.target_strings
        ]
        best = max(measurements, key=lambda item: item["projection"])
        return {
            **best,
            "all_targets": measurements,
        }

    def generate_response(self, prompt: str, max_new_tokens: int = 40) -> tuple[str, torch.Tensor]:
        suffix_str = self.decode_suffix()
        full_user_content = prompt if not suffix_str.strip() else prompt + " " + suffix_str
        formatted_prompt = self.model.format_chat(full_user_content, assistant_prefix=None)
        inputs = self.model.tokenize(formatted_prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.model.decode(outputs[0, input_len:]).strip()
        return response, inputs["input_ids"]

    def track_generation_trajectory(
        self,
        prompt: str,
        checkpoints: tuple[int, ...] = (1, 3, 5, 10),
        max_new_tokens: int = 40,
    ) -> dict:
        suffix_str = self.decode_suffix()
        full_user_content = prompt if not suffix_str.strip() else prompt + " " + suffix_str
        formatted_prompt = self.model.format_chat(full_user_content, assistant_prefix=None)
        inputs = self.model.tokenize(formatted_prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        generated_tokens = []
        checkpoint_set = set(checkpoints)
        trajectory = []

        for _ in range(max_new_tokens + 1):
            with torch.no_grad():
                outputs = self.raw_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            generated_count = len(generated_tokens)
            if generated_count in checkpoint_set:
                hidden_state = outputs.hidden_states[self.best_layer + 1][0, -1, :].float()
                is_compliance, proj_val, status, decision = self.classify_activation_vector(hidden_state)
                trajectory.append({
                    "step": generated_count,
                    "status": status,
                    "is_compliance": is_compliance,
                    "projection": proj_val,
                    "decision": decision,
                    "token": self.model.decode(torch.tensor([generated_tokens[-1]], device=input_ids.device)).strip()
                    if generated_tokens else "",
                    "text_so_far": self.model.decode(
                        torch.tensor(generated_tokens, device=input_ids.device),
                    ).strip(),
                })

            if generated_count >= max_new_tokens:
                break

            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_val = next_token.item()
            if token_val == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_val)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            ], dim=-1)

        response = self.model.decode(
            torch.tensor(generated_tokens, device=input_ids.device),
        ).strip() if generated_tokens else ""
        generated_any_compliance = any(item["is_compliance"] for item in trajectory)
        return {
            "trajectory": trajectory,
            "response": response,
            "generated_any_compliance": generated_any_compliance,
        }

    def check_success(self, prompt: str) -> tuple[bool, str]:
        """
        Checks objective-side forced-target activation success.

        This is intentionally separate from behavioral jailbreak success during
        free generation, which is tracked in `generated_any_compliance`.
        """
        forced = self.measure_forced_targets(prompt)
        generated = self.track_generation_trajectory(prompt, max_new_tokens=40)
        behavior = score_response(prompt, generated["response"])
        gate_passed = behavior_gate_passed(behavior)

        proj_str = f"proj={forced['projection']:+.1f}"
        label = behavior["label"]
        print(
            f"    [ForcedTarget] {forced['status'].upper()} ({proj_str}) "
            f"| [GeneratedAny] {generated['generated_any_compliance']} "
            f"| [Behavior] {label}"
        )

        objective_success = forced["is_compliance"]
        early_stop_success = (
            objective_success and gate_passed
            if self.require_behavior_for_early_stop
            else objective_success
        )

        return early_stop_success, generated["response"], forced, generated, behavior, gate_passed

    def _make_result(
        self,
        *,
        steps: int,
        response: str,
        losses: list[float],
        target_losses: list[float],
        activation_losses: list[float],
        target_strings_used: list[str],
        forced: dict,
        generated: dict,
        behavior: dict,
        gate_passed: bool,
        prune_refine: dict | None = None,
    ) -> dict:
        forced_success = bool(forced["is_compliance"])
        return {
            "forced_target_success": forced_success,
            "success": forced_success,
            "behavior_gate_passed": gate_passed,
            "steps": steps,
            "suffix": self.decode_suffix(),
            "suffix_token_count": int(self.suffix_length),
            "response": response,
            "losses": list(losses),
            "target_losses": list(target_losses),
            "activation_losses": list(activation_losses),
            "target_strings": self.target_strings,
            "target_strings_used": list(target_strings_used),
            "forced_target": forced,
            "generation_trajectory": generated,
            "behavior_score": behavior,
            "candidate_archive": list(self.candidate_archive),
            "prune_refine": prune_refine or {"enabled": False},
            "loss_behavior_gap": forced_success and not gate_passed,
        }

    def _prune_suffix_tokens(self, prompt: str, target_string: str) -> dict:
        initial_suffix = self.suffix_ids.clone()
        initial_length = len(initial_suffix)
        target_length = prune_target_length(
            initial_length,
            self.prune_fraction,
            self.prune_min_tokens,
        )
        summary = {
            "enabled": True,
            "attempted": True,
            "accepted": False,
            "initial_token_count": int(initial_length),
            "target_token_count": int(target_length),
            "pruned_token_count": int(initial_length),
            "removed_tokens": [],
            "refine_steps": 0,
        }

        if target_length >= initial_length:
            summary["attempted"] = False
            summary["reason"] = "target length is not shorter than current suffix"
            return summary

        current = initial_suffix
        losses, _, _ = self._evaluate_suffix_candidates(prompt, target_string, [current])
        current_loss = losses[0]
        summary["initial_loss"] = float(current_loss)

        while len(current) > target_length:
            candidates = []
            metadata = []
            for pos in range(len(current)):
                candidate = torch.cat([current[:pos], current[pos + 1:]])
                candidates.append(candidate)
                metadata.append({
                    "position": int(pos),
                    "token": self._decode_ids(current[pos:pos + 1]),
                })

            candidate_losses, _, _ = self._evaluate_suffix_candidates(
                prompt,
                target_string,
                candidates,
            )
            best_idx = min(range(len(candidate_losses)), key=lambda idx: candidate_losses[idx])
            best_loss = candidate_losses[best_idx]
            tolerance = abs(current_loss) * self.prune_max_rel_loss_increase

            if best_loss > current_loss + tolerance:
                summary["stop_reason"] = "loss increase exceeded tolerance"
                summary["final_loss"] = float(current_loss)
                break

            removed = metadata[best_idx]
            removed["loss_before"] = float(current_loss)
            removed["loss_after"] = float(best_loss)
            summary["removed_tokens"].append(removed)
            current = candidates[best_idx]
            current_loss = best_loss
        else:
            summary["stop_reason"] = "target length reached"
            summary["final_loss"] = float(current_loss)

        if len(current) < initial_length:
            self.suffix_ids = current.to(self.raw_model.device)
            self.suffix_length = len(current)
            self._momentum_grad = None
            self._seen_suffixes.add(tuple(self.suffix_ids.tolist()))

        summary["pruned_token_count"] = int(len(current))
        summary["pruned"] = len(current) < initial_length
        return summary

    def _maybe_prune_and_refine(
        self,
        prompt: str,
        *,
        steps: int,
        response: str,
        losses: list[float],
        target_losses: list[float],
        activation_losses: list[float],
        target_strings_used: list[str],
        forced: dict,
        generated: dict,
        behavior: dict,
        gate_passed: bool,
    ) -> dict:
        original_result = self._make_result(
            steps=steps,
            response=response,
            losses=losses,
            target_losses=target_losses,
            activation_losses=activation_losses,
            target_strings_used=target_strings_used,
            forced=forced,
            generated=generated,
            behavior=behavior,
            gate_passed=gate_passed,
        )

        if (
            not self.enable_prune_refine
            or self.prune_fraction <= 0
            or self.suffix_length <= self.prune_min_tokens
        ):
            original_result["prune_refine"] = {
                "enabled": self.enable_prune_refine,
                "attempted": False,
            }
            return original_result

        original_suffix = self.suffix_ids.clone()
        original_suffix_length = self.suffix_length
        original_momentum = None if self._momentum_grad is None else self._momentum_grad.clone()

        target_string = forced.get("target_string", self.target_string)
        prune_summary = self._prune_suffix_tokens(prompt, target_string)
        if not prune_summary.get("pruned"):
            original_result["prune_refine"] = prune_summary
            return original_result

        print(
            f"  [Prune] {prune_summary['initial_token_count']} -> "
            f"{prune_summary['pruned_token_count']} tokens "
            f"({len(prune_summary['removed_tokens'])} removed)"
        )

        refine_steps_run = 0
        for refine_idx in range(1, self.prune_refine_steps + 1):
            refine_steps_run = refine_idx
            step_res = self.step(prompt, step_idx=steps + refine_idx)
            losses.append(step_res["loss"])
            target_losses.append(step_res["target_loss"])
            activation_losses.append(step_res["activation_loss"])
            target_strings_used.append(step_res["target_string"])

            if refine_idx == 1 or refine_idx % 10 == 0 or refine_idx == self.prune_refine_steps:
                suffix_safe = safe_console_text(step_res["suffix"], 40)
                print(
                    f"  Prune Refine {refine_idx:03d} | Loss: {step_res['loss']:.4f} "
                    f"| CE: {step_res['target_loss']:.4f} "
                    f"| Act: {step_res['activation_loss']:.4f} "
                    f"| Suffix: {suffix_safe}..."
                )

        prune_summary["refine_steps"] = refine_steps_run
        steps += refine_steps_run
        _, response, forced, generated, behavior, gate_passed = self.check_success(prompt)
        candidate_result = self._make_result(
            steps=steps,
            response=response,
            losses=losses,
            target_losses=target_losses,
            activation_losses=activation_losses,
            target_strings_used=target_strings_used,
            forced=forced,
            generated=generated,
            behavior=behavior,
            gate_passed=gate_passed,
            prune_refine=prune_summary,
        )

        original_ok = (
            original_result["forced_target_success"]
            and (
                original_result["behavior_gate_passed"]
                or not self.require_behavior_for_early_stop
            )
        )
        candidate_ok = (
            candidate_result["forced_target_success"]
            and (
                candidate_result["behavior_gate_passed"]
                or not self.require_behavior_for_early_stop
            )
        )

        if candidate_ok or (not original_ok and candidate_result["forced_target_success"]):
            prune_summary["accepted"] = True
            candidate_result["prune_refine"] = prune_summary
            return candidate_result

        self.suffix_ids = original_suffix
        self.suffix_length = original_suffix_length
        self._momentum_grad = original_momentum
        prune_summary["accepted"] = False
        prune_summary["reject_reason"] = "pruned suffix failed final forced/behavior acceptance checks"
        original_result["prune_refine"] = prune_summary
        return original_result

    def optimize(self, prompt: str, max_steps=500, check_interval=25) -> dict:
        """
        Full optimization loop for a prompt.
        """
        n_layers = len(self.direction_layers)
        print(f"\n[*] Optimizing suffix for prompt: {safe_console_text(prompt, 60)}...")
        print(f"    Multi-layer GCG across {n_layers} layers: L{self.direction_layers[0]}-L{self.direction_layers[-1]}")
        
        losses = []
        target_losses = []
        activation_losses = []
        target_strings_used = []
        completed_steps = 0
        final_response = None
        final_forced = None
        final_generated = None
        final_behavior = None
        final_gate_passed = False
        
        for step_idx in range(1, max_steps + 1):
            step_res = self.step(prompt, step_idx=step_idx)
            completed_steps = step_idx
            loss_val = step_res["loss"]
            losses.append(loss_val)
            target_losses.append(step_res["target_loss"])
            activation_losses.append(step_res["activation_loss"])
            target_strings_used.append(step_res["target_string"])
            
            if step_idx % 10 == 0 or step_idx == 1:
                suffix_safe = safe_console_text(step_res["suffix"], 40)
                print(
                    f"  Step {step_idx:03d} | Loss: {loss_val:.4f} "
                    f"| CE: {step_res['target_loss']:.4f} "
                    f"| Act: {step_res['activation_loss']:.4f} "
                    f"| Target: {safe_console_text(step_res['target_string'], 24)} "
                    f"| Suffix: {suffix_safe}..."
                )
                
            if step_idx % check_interval == 0:
                early_stop_success, response, forced, generated, behavior, gate_passed = self.check_success(prompt)
                forced_success = bool(forced["is_compliance"])
                resp_safe = safe_console_text(response, 50)
                print(
                    f"  [Check] Forced-target success: {forced_success} "
                    f"| Behavior gate: {gate_passed} | Response snippet: {resp_safe}..."
                )
                if early_stop_success:
                    stop_reason = (
                        "forced-target + behavior gate"
                        if self.require_behavior_for_early_stop
                        else "forced-target activation success"
                    )
                    print(f"[+] Early stopping at step {step_idx}: {stop_reason}.")
                    final_response = response
                    final_forced = forced
                    final_generated = generated
                    final_behavior = behavior
                    final_gate_passed = gate_passed
                    break

                    
        if final_response is None:
            _, final_response, final_forced, final_generated, final_behavior, final_gate_passed = (
                self.check_success(prompt)
            )

        return self._maybe_prune_and_refine(
            prompt,
            steps=completed_steps,
            response=final_response,
            losses=losses,
            target_losses=target_losses,
            activation_losses=activation_losses,
            target_strings_used=target_strings_used,
            forced=final_forced,
            generated=final_generated,
            behavior=final_behavior,
            gate_passed=final_gate_passed,
        )
