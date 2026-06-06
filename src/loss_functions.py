import torch
import torch.nn.functional as F

def compute_loss(logits, hidden_states, input_ids, suffix_slice, target_slice,
                 direction_vecs, direction_layers, alpha=1.0, beta=0.3,
                 aggregation='mean'):
    """
    Computes the combined multi-layer loss for GCG.
    
    L = alpha * L_target + beta * L_activation
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        hidden_states: tuple of tensors from output_hidden_states=True
        input_ids: Tensor of shape [batch_size, seq_len] or [seq_len]
        suffix_slice: slice object indicating the positions of the suffix tokens
        target_slice: slice object indicating the positions of the target tokens
        direction_vecs: dict mapping layer_idx -> direction Tensor of shape [hidden_dim]
        direction_layers: list of int, the target layer indices (e.g. [8,9,10,11,12,13,14])
        alpha: float, weight for target CE loss
        beta: float, weight for activation projection loss
        aggregation: str, how to aggregate per-layer activation losses:
            'mean' - average across layers (for gradient computation, all layers contribute)
            'max'  - worst layer determines the loss (minimax, for candidate selection:
                     forces ALL layers to converge toward target simultaneously)
    """
    batch_size = logits.shape[0]
    
    # 1. Target Loss (Cross-Entropy)
    loss_start = target_slice.start - 1
    loss_stop = target_slice.stop - 1
    
    target_logits = logits[:, loss_start:loss_stop, :]
    
    if input_ids.dim() == 1:
        target_ids = input_ids[target_slice].unsqueeze(0).expand(batch_size, -1)
    else:
        target_ids = input_ids[:, target_slice]
        
    loss_target = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none"
    )
    target_len = target_slice.stop - target_slice.start
    loss_target = loss_target.view(batch_size, target_len).mean(dim=1)
    
    # 2. Multi-Layer Activation Loss
    # Match the direction extraction position: the hidden state after the model
    # has produced the response prefix, not the pre-generation prompt state.
    activation_pos = target_slice.stop - 1
    
    layer_losses = []
    for layer_idx in direction_layers:
        # Output of layers[idx] is hidden_states[idx + 1]
        layer_output = hidden_states[layer_idx + 1]  # [batch_size, seq_len, hidden_dim]
        last_token_acts = layer_output[:, activation_pos, :]  # [batch_size, hidden_dim]
        
        d_vec = direction_vecs[layer_idx].to(last_token_acts.device).to(last_token_acts.dtype)
        
        # Dot product: [batch_size]
        projections = torch.einsum("bd,d->b", last_token_acts, d_vec)
        
        # Negative projection (we want to maximize projection = minimize negative)
        layer_losses.append(-projections)  # [batch_size]
    
    # Aggregate across layers: [num_layers, batch_size] -> [batch_size]
    stacked_layer_losses = torch.stack(layer_losses)  # [num_layers, batch_size]
    
    if aggregation == 'max':
        # Minimax: the WORST-performing layer determines the loss.
        # This forces candidate selection to pick tokens that improve ALL layers,
        # preventing the optimizer from sacrificing some layers for others.
        loss_activation = stacked_layer_losses.max(dim=0).values
    else:
        # Mean: all layers contribute equally (good for gradient computation)
        loss_activation = stacked_layer_losses.mean(dim=0)
    
    # Combined loss per batch item: [batch_size]
    combined_loss = alpha * loss_target + beta * loss_activation
    
    return combined_loss, loss_target, loss_activation
