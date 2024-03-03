import torch

def get_one_hot_channels(seg, indices):
    """
    Parameters:
    - seg: [H, W] tensor with integers
    - indices: [M] tensor with selected indices for one-hot encoding.
    - N: Number of classes (int).
    
    Returns:
    - [H, W, M] tensor representing the one-hot encoded segmentation map for selected indices.
    """
    H, W = seg.shape
    M = len(indices)
    
    seg_expanded = seg.unsqueeze(-1).expand(H, W, M)
    indices_expanded = indices.expand(H, W, M)
    output = (seg_expanded == indices_expanded)
    return output

def one_hot_to_integer(one_hot_mask):
    values, indices = one_hot_mask.max(dim=-1)
    return torch.where(values > 0, indices, torch.tensor(-1))

def integer_to_one_hot(int_tensor, num_classes):
    mask = (int_tensor >= 0) & (int_tensor < num_classes)
    int_tensor = torch.where(mask, int_tensor, torch.tensor(0))
    one_hot = torch.nn.functional.one_hot(int_tensor, num_classes)
    one_hot = torch.where(mask.unsqueeze(-1), one_hot, False)
    return one_hot

def maybe_convert_to_one_hot(batch, cfg=None, num_classes=None):
    num_classes = num_classes or cfg.model.num_token_cls + 1
    if batch.gen_segmentation.ndim == 3:
        batch.gen_segmentation = integer_to_one_hot(batch.gen_segmentation, num_classes)
    if batch.disc_segmentation.ndim == 3:
        batch.disc_segmentation = integer_to_one_hot(batch.disc_segmentation, num_classes)

    return batch

