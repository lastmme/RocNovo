import torch

def encoder_causal_mask(size: int, device=None):
    # 返回 shape (size, size)
    mask = torch.full((size, size), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask