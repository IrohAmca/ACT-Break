import torch

class ActivationSteerer:
    """Modifies multiple target layers' hidden states during forward pass by adding steering vectors."""
    def __init__(self, hooked_model, direction_vecs: dict, layer_indices: list[int]):
        """
        Args:
            hooked_model: HookedModel instance
            direction_vecs: dict mapping layer_idx -> direction Tensor [hidden_dim]
            layer_indices: list of layer indices to steer (e.g. [8, 9, 10, 11, 12, 13, 14])
        """
        self.hooked_model = hooked_model
        self.direction_vecs = direction_vecs
        self.layer_indices = layer_indices
        self.hook_handles = []

    def register_hooks(self, alpha: float):
        """Registers forward hooks at all target layers that add alpha * direction_vec."""
        self.remove_hooks()
        
        device = self.hooked_model.model.device
        dtype = self.hooked_model.model.dtype
        layers = (
            self.hooked_model._layers
            if getattr(self.hooked_model, "_layers", None) is not None
            else self.hooked_model._resolve_transformer_layers()
        )
        
        for layer_idx in self.layer_indices:
            d_vec = self.direction_vecs[layer_idx]
            steer_add = (d_vec.to(device).to(dtype) * alpha)
            
            layer = layers[layer_idx]
            
            def hook_fn(module, input, output, _steer=steer_add):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    new_hidden = hidden_states + _steer
                    return (new_hidden,) + output[1:]
                else:
                    return output + _steer
                    
            handle = layer.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Removes all active steering hooks."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def steer_and_generate(self, prompt: str, alpha: float, max_new_tokens: int = 50) -> str:
        """Helper to format prompt, register hooks on all layers, generate text, and clean up."""
        self.register_hooks(alpha)
        try:
            formatted_prompt = self.hooked_model.format_chat(prompt, assistant_prefix=None)
            inputs = self.hooked_model.tokenize(formatted_prompt)
            with torch.no_grad():
                outputs = self.hooked_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens
                )
            
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = outputs[0, input_len:]
            response = self.hooked_model.decode(gen_tokens)
            return response.strip()
        finally:
            self.remove_hooks()

    def sweep_alpha(self, prompt: str, alphas: list[float], max_new_tokens: int = 50) -> dict[float, str]:
        """Runs steering generation over a list of alpha values."""
        results = {}
        for alpha in alphas:
            results[alpha] = self.steer_and_generate(prompt, alpha, max_new_tokens=max_new_tokens)
        return results
