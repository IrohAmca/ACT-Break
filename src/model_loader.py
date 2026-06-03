import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationStore:
    """Stores activations captured at forward hooks."""
    def __init__(self):
        self.activations: dict[int, torch.Tensor] = {}

    def clear(self):
        self.activations.clear()

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self.activations.get(layer_idx)

class HookedModel:
    """Wrapper around Qwen model to register forward hooks and extract hidden states."""
    def __init__(self, model_name: str, target_layers: list[int], dtype: str = "float16", device: str = "cuda"):
        self.model_name = model_name
        self.target_layers = target_layers
        self.dtype = getattr(torch, dtype)
        self.device = device

        self.model = None
        self.tokenizer = None
        self.store = ActivationStore()
        self._hooks = []

    def load(self):
        print(f"[*] Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
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
        self.model.generation_config.do_sample = False
        for sample_only_flag in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, sample_only_flag):
                setattr(self.model.generation_config, sample_only_flag, None)

        self._register_hooks()
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[+] Loaded {self.model_name} ({params:.0f}M parameters)")
        return self

    def _register_hooks(self):
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]

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

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(assistant_prefix is None),
        )

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
