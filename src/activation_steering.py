import torch

class ActivationSteerer:
    """Modifies the target layer's hidden states during the forward pass by adding a steering vector."""
    def __init__(self, hooked_model, direction_vec: torch.Tensor, layer_idx: int):
        self.hooked_model = hooked_model
        self.direction_vec = direction_vec
        self.layer_idx = layer_idx
        self.hook_handle = None

    def register_hook(self, alpha: float):
        """Registers a forward hook at layer_idx that adds alpha * direction_vec to the layer's output."""
        self.remove_hook()
        
        # Bring vector to appropriate device and dtype
        device = self.hooked_model.model.device
        dtype = self.hooked_model.model.dtype
        # We need to unsqueeze or scale
        # direction_vec is shape [hidden_dim]
        steer_add = (self.direction_vec.to(device).to(dtype) * alpha)
        
        layer = self.hooked_model.model.model.layers[self.layer_idx]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Avoid in-place modification to prevent side-effects/errors
                new_hidden = hidden_states + steer_add
                return (new_hidden,) + output[1:]
            else:
                return output + steer_add
                
        self.hook_handle = layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        """Removes the active steering hook if present."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def steer_and_generate(self, prompt: str, alpha: float, max_new_tokens: int = 50) -> str:
        """Helper to format prompt, register hook, generate text, and clean up."""
        self.register_hook(alpha)
        try:
            formatted_prompt = self.hooked_model.format_chat(prompt, assistant_prefix=None)
            inputs = self.hooked_model.tokenize(formatted_prompt)
            with torch.no_grad():
                outputs = self.hooked_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens
                )
            
            # Extract generated tokens only (strip input tokens)
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = outputs[0, input_len:]
            response = self.hooked_model.decode(gen_tokens)
            return response.strip()
        finally:
            self.remove_hook()

    def sweep_alpha(self, prompt: str, alphas: list[float], max_new_tokens: int = 50) -> dict[float, str]:
        """Runs steering generation over a list of alpha values."""
        results = {}
        for alpha in alphas:
            results[alpha] = self.steer_and_generate(prompt, alpha, max_new_tokens=max_new_tokens)
        return results
