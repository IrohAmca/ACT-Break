import torch
import torch.nn.functional as F

def compute_loss(logits, hidden_states, input_ids, suffix_slice, target_slice,
                 direction_vec, direction_layer, alpha=1.0, beta=0.3):
    """
    Computes the combined loss for GCG.
    
    L = alpha * L_target + beta * L_activation
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        hidden_states: tuple of tensors, where hidden_states[direction_layer + 1] has shape [batch_size, seq_len, hidden_dim]
        input_ids: Tensor of shape [batch_size, seq_len] or [seq_len]
        suffix_slice: slice object indicating the positions of the suffix tokens
        target_slice: slice object indicating the positions of the target tokens
        direction_vec: Tensor of shape [hidden_dim]
        direction_layer: int, the index of the target layer (e.g. 12)
        alpha: float, weight for target CE loss
        beta: float, weight for activation projection loss
    """
    batch_size = logits.shape[0]
    
    # 1. Target Loss (Cross-Entropy)
    # The targets are at target_slice.
    # The logits that predict them are shifted by 1 to the left (i.e. target_slice.start - 1 to target_slice.stop - 1)
    loss_start = target_slice.start - 1
    loss_stop = target_slice.stop - 1
    
    # Extract logits for target tokens: [batch_size, target_len, vocab_size]
    target_logits = logits[:, loss_start:loss_stop, :]
    
    # Get target token IDs
    if input_ids.dim() == 1:
        target_ids = input_ids[target_slice].unsqueeze(0).expand(batch_size, -1)
    else:
        target_ids = input_ids[:, target_slice]
        
    # Reshape for cross_entropy: input is [N, C], target is [N]
    loss_target = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none"
    )
    # Average per batch item
    target_len = target_slice.stop - target_slice.start
    loss_target = loss_target.view(batch_size, target_len).mean(dim=1)
    
    # 2. Activation Loss
    # We want to pull target layer's output at suffix positions in the direction of direction_vec.
    # L_activation = -mean(hidden_states[direction_layer][:, suffix_positions, :] @ V_jailbreak)
    # Output of layers[idx] is hidden_states[idx + 1].
    layer_output = hidden_states[direction_layer + 1] # shape [batch_size, seq_len, hidden_dim]
    
    # Suffix activations: [batch_size, suffix_len, hidden_dim]
    suffix_acts = layer_output[:, suffix_slice, :]
    
    # Project onto direction_vec. direction_vec has shape [hidden_dim]
    d_vec = direction_vec.to(suffix_acts.device).to(suffix_acts.dtype)
    
    # Dot product/projection per token: [batch_size, suffix_len]
    projections = torch.einsum("bsd,d->bs", suffix_acts, d_vec)
    
    # Minimize negative projection (maximize projection)
    loss_activation = -projections.mean(dim=1)
    
    # Combined loss per batch item: [batch_size]
    combined_loss = alpha * loss_target + beta * loss_activation
    
    return combined_loss, loss_target, loss_activation
