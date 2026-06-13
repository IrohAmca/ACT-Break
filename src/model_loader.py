import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

class ActivationStore:
    """Stores activations captured at forward hooks."""
    def __init__(self):
        self.activations: dict[int, torch.Tensor] = {}

    def clear(self):
        self.activations.clear()

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self.activations.get(layer_idx)

class HookedModel:
    """Wrapper around causal LMs to register forward hooks and extract hidden states."""
    def __init__(self, model_name: str, target_layers: list[int], dtype: str = "float16", device: str = "cuda"):
        self.model_name = model_name
        self.target_layers = target_layers
        self.dtype = getattr(torch, dtype)
        self.device = device

        self.model = None
        self.tokenizer = None
        self.store = ActivationStore()
        self._hooks = []
        self._layers = None

    def load(self):
        print(f"[*] Loading model: {self.model_name}")
        self.tokenizer = self._load_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": self.device,
            "trust_remote_code": True,
        }
        transformers_major = int(transformers.__version__.split(".", 1)[0])
        dtype_key = "dtype" if transformers_major >= 5 else "torch_dtype"
        model_kwargs[dtype_key] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
        self._layers = self._resolve_transformer_layers()
        self._validate_target_layers(self._layers)
        self.model.generation_config.do_sample = False
        for sample_only_flag in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, sample_only_flag):
                setattr(self.model.generation_config, sample_only_flag, None)

        self._register_hooks()
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[+] Loaded {self.model_name} ({params:.0f}M parameters)")
        print(f"[+] Hidden layers: {len(self._layers)} | target layers: {self.target_layers}")
        return self

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
        except ValueError as exc:
            message = str(exc)
            if "Tokenizer class" not in message or "does not exist" not in message:
                raise

            print("[!] AutoTokenizer could not resolve the tokenizer class.")
            print("[!] Falling back to PreTrainedTokenizerFast from tokenizer.json.")
            return PreTrainedTokenizerFast.from_pretrained(
                self.model_name,
                padding_side="left",
            )

    def _resolve_transformer_layers(self):
        layer_paths = [
            ("model.layers", lambda model: model.model.layers),
            ("transformer.h", lambda model: model.transformer.h),
            ("gpt_neox.layers", lambda model: model.gpt_neox.layers),
        ]
        for _path, getter in layer_paths:
            try:
                layers = getter(self.model)
            except AttributeError:
                continue
            if layers is not None:
                return layers

        model_type = getattr(getattr(self.model, "config", None), "model_type", "unknown")
        raise TypeError(
            f"Could not resolve transformer layers for model_type={model_type!r}. "
            "Add the model's layer path to HookedModel._resolve_transformer_layers()."
        )

    def _validate_target_layers(self, layers):
        if not self.target_layers:
            raise ValueError("target_layers cannot be empty.")

        max_layer = len(layers) - 1
        invalid_layers = [
            layer_idx
            for layer_idx in self.target_layers
            if layer_idx < 0 or layer_idx > max_layer
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid target layers for {self.model_name}: {invalid_layers}. "
                f"Model has {len(layers)} layers, valid range is 0-{max_layer}."
            )

    def _register_hooks(self):
        layers = self._layers if self._layers is not None else self._resolve_transformer_layers()
        for layer_idx in self.target_layers:
            layer = layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self.store.activations[idx] = hidden_states.detach().cpu()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def format_chat(self, user_message: str, assistant_prefix: str | None = None) -> str:
        messages = [{"role": "user", "content": user_message}]
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(assistant_prefix is None),
            )

        if assistant_prefix:
            return f"User: {user_message}\nAssistant: {assistant_prefix}"
        return f"User: {user_message}\nAssistant:"

    def tokenize(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", padding=False)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 20,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
