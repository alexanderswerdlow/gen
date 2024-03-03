import torch

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
    if batch["gen_segmentation"].ndim == 3:
        batch["gen_segmentation"] = integer_to_one_hot(batch["gen_segmentation"], num_classes)
    if batch['disc_segmentation'].ndim == 3:
        batch['disc_segmentation'] = integer_to_one_hot(batch['disc_segmentation'], num_classes)

    return batch