import torch
from src.loss_functions import compute_loss

def compute_token_gradients(model, prefix_ids, suffix_ids, target_ids,
                            direction_vec, direction_layer,
                            alpha=1.0, beta=0.3):
    """
    Computes gradients of the loss with respect to the one-hot representation of suffix tokens.
    
    Args:
        model: HookedModel or the raw AutoModelForCausalLM.
        prefix_ids: 1D Tensor of prefix token IDs
        suffix_ids: 1D Tensor of current suffix token IDs
        target_ids: 1D Tensor of target token IDs
        direction_vec: Tensor of shape [hidden_dim]
        direction_layer: int, layer to target
        alpha: float, weight for target CE loss
        beta: float, weight for activation projection loss
        
    Returns:
        Tensor of shape [suffix_len, vocab_size] representing the gradient of the loss
        with respect to the one-hot representation of the suffix tokens.
    """
    # Extract underlying HF model
    raw_model = model.model if hasattr(model, "model") else model
    
    # Device and dtype
    device = raw_model.device
    dtype = raw_model.dtype
    
    prefix_ids = prefix_ids.to(device)
    suffix_ids = suffix_ids.to(device)
    target_ids = target_ids.to(device)
    
    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)
    target_len = len(target_ids)
    
    # Slice indices in the full sequence
    suffix_slice = slice(prefix_len, prefix_len + suffix_len)
    target_slice = slice(prefix_len + suffix_len, prefix_len + suffix_len + target_len)
    
    # Get embedding matrix
    embed_tokens = raw_model.model.embed_tokens
    embedding_matrix = embed_tokens.weight # [vocab_size, hidden_dim]
    vocab_size = embedding_matrix.shape[0]
    
    # Create differentiable one-hot representation of suffix
    one_hot = torch.zeros(suffix_len, vocab_size, device=device, dtype=dtype)
    one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_()
    
    # Suffix embeddings: [suffix_len, hidden_dim]
    suffix_embeds = one_hot @ embedding_matrix
    
    # Prefix and target embeddings
    with torch.no_grad():
        prefix_embeds = embed_tokens(prefix_ids) # [prefix_len, hidden_dim]
        target_embeds = embed_tokens(target_ids) # [target_len, hidden_dim]
        
    # Concatenate embeddings: [1, seq_len, hidden_dim]
    full_embeds = torch.cat([
        prefix_embeds.unsqueeze(0),
        suffix_embeds.unsqueeze(0),
        target_embeds.unsqueeze(0)
    ], dim=1)
    
    # Full IDs (for loss target identification)
    full_ids = torch.cat([prefix_ids, suffix_ids, target_ids], dim=0)
    
    # Forward pass
    outputs = raw_model(inputs_embeds=full_embeds, output_hidden_states=True)
    
    # Compute combined loss
    loss, _, _ = compute_loss(
        logits=outputs.logits,
        hidden_states=outputs.hidden_states,
        input_ids=full_ids,
        suffix_slice=suffix_slice,
        target_slice=target_slice,
        direction_vec=direction_vec,
        direction_layer=direction_layer,
        alpha=alpha,
        beta=beta
    )
    
    # Backward pass to get gradients
    loss.backward()
    
    grad = one_hot.grad.clone()
    
    # Zero out gradients for safety
    raw_model.zero_grad()
    
    return grad
