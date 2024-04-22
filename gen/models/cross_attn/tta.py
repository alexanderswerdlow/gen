from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from image_utils.standalone_image_utils import integer_to_color
import numpy as np
import torch
import torch.nn.functional as F
from gen.utils.data_defs import undo_normalization_given_transforms
from gen.utils.logging_utils import log_info
from gen.utils.trainer_utils import TrainingState, unwrap
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from image_utils import Im, onehot_to_color
from gen.utils.data_defs import integer_to_one_hot, undo_normalization_given_transforms


if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData


def print_tensor_attributes(tensor: torch.Tensor) -> None:

    print(f"requires_grad: {tensor.requires_grad}, "
          f"grad_fn: {tensor.grad_fn is not None}, "
          f"is_leaf: {tensor.is_leaf}, "
          f"grad: {tensor.grad is not None}")

def tta_inference(
    self: BaseMapper, 
    batch: InputData, 
    state: TrainingState,
    accelerator: Accelerator,
    num_optim_steps: int = 15,
    bs: int = 8,
    gradient_accumulation_steps: int = 16,
):
    unwrap(self).set_inference_mode()
    self.eval()

    initial_prompt_images, _ = self.infer_batch(batch=batch, num_images_per_prompt=2)
    
    batch.force_use_orig_src_tokens = True
    batch.force_encode_all_masks = torch.full((batch.bs,), True, device=batch.src_segmentation.device)
    assert self.training is False

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            cond = self.get_standard_conditioning_for_inference(batch=batch.clone())
            
    batch.force_use_orig_src_tokens = False
    batch.force_repeat_cond_data = True

    input_batch = batch.clone().expand(bs)
    num_new_tokens = 8
    num_tokens = min(cond.gt_src_mask_token.shape[0], self.cfg.model.num_decoder_cross_attn_tokens - num_new_tokens)

    assert torch.allclose(cond.encoder_hidden_states[0, :num_tokens], cond.gt_src_mask_token[:num_tokens])
    torch.set_grad_enabled(True)

    orig_tokens = cond.gt_src_mask_token.clone()
    input_tensor = cond.gt_src_mask_token[:num_tokens].to(torch.float32).detach().requires_grad_(True)
    input_tensor_alphas = torch.ones((input_tensor.shape[0]), device=input_tensor.device).detach().requires_grad_(True)
    
    new_tokens = (1e-2 * torch.randn((num_new_tokens, input_tensor.shape[1]), device=batch.device)).detach().requires_grad_(True)
    new_tokens_alphas = (torch.zeros((num_new_tokens,), device=batch.device) + 1e-8).detach().requires_grad_(True)

    dummy_tokens = cond.encoder_hidden_states[:, num_tokens:].detach()
    assert self.uncond_hidden_states.sum() == 0
    
    print_tensor_attributes(input_tensor)
    print_tensor_attributes(input_tensor_alphas)

    def get_params():
        return [input_tensor, input_tensor_alphas, new_tokens, new_tokens_alphas]

    import bitsandbytes as bnb
    
    sgd = False
    optimizer_cls = torch.optim.SGD if sgd else bnb.optim.AdamW8bit
    opt_params = dict(lr=1e-3, momentum=0.001) if sgd else dict(lr=5e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    optimizer = optimizer_cls(
        get_params(),
        **opt_params
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_optim_steps,
    )

    self, lr_scheduler, optimizer = accelerator.prepare(self, lr_scheduler, optimizer)
    forward_steps = 0

    for i in range(num_optim_steps):
        total_loss = 0
        for j in range(gradient_accumulation_steps):
            cond.encoder_hidden_states = torch.cat((
                input_tensor[None],
                new_tokens[None],
                torch.zeros((1, self.cfg.model.num_decoder_cross_attn_tokens - (num_tokens + num_new_tokens), input_tensor.shape[-1]), device=batch.device)
            ), dim=1)
            cond.encoder_hidden_states[:, :num_tokens] *= input_tensor_alphas[None, :, None]
            cond.encoder_hidden_states[:, num_tokens:num_tokens + num_new_tokens] *= new_tokens_alphas[None, :, None]
            cond.encoder_hidden_states = cond.encoder_hidden_states.repeat(bs, 1, 1)
            cond.unet_kwargs = dict()
            losses = self(batch=input_batch, state=state, cond=cond)
            old_token_regularization_loss = 1e-4 * torch.norm(1 - input_tensor_alphas, 1)
            new_token_regularization_loss = 1e-4 * torch.norm(new_tokens_alphas, 1)
            loss = losses['diffusion_loss'] + old_token_regularization_loss + new_token_regularization_loss
            total_loss += loss.detach()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if j == (gradient_accumulation_steps - 1):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                print(f"Backward Step: {i}, loss: {loss.item():.4f}, all loss: {total_loss.item():.5f}, new tok sum: {new_tokens.sum().item():.4f}, in alphas: {input_tensor_alphas.min().item():.4f}, new_alphas: {new_tokens.max().item():.4f}, {cond.encoder_hidden_states.sum().item():.5f}, LR: {lr_scheduler.get_last_lr()[0]:.5f} Diff: {losses['diffusion_loss'].item():.4f} Old Tok Loss: {old_token_regularization_loss.item():.4f} New Tok Loss: {new_token_regularization_loss.item():.4f}")

            forward_steps += 1

    _orig_src_one_hot = integer_to_one_hot(batch.src_segmentation.clone())
    _conditioned_src_seg = batch.src_segmentation.clone()
    _first_valid_seg = torch.isin(_conditioned_src_seg, cond.mask_instance_idx)
    _conditioned_src_seg[~_first_valid_seg] = 255
    _first_src_one_hot__ = integer_to_one_hot(_conditioned_src_seg)

    cond.encoder_hidden_states = torch.cat((
                input_tensor[None],
                new_tokens[None],
                torch.zeros((1, self.cfg.model.num_decoder_cross_attn_tokens - (num_tokens + num_new_tokens), input_tensor.shape[-1]), device=batch.device)
    ), dim=1)
    cond.unet_kwargs = dict(prompt_embeds=cond.encoder_hidden_states, cross_attention_kwargs=dict())
    optimized_images, _ = self.infer_batch(batch=batch, cond=cond, num_images_per_prompt=2, allow_get_cond=False)

    orig_src_image = Im(undo_normalization_given_transforms(self.cfg.dataset.val.augmentation.src_transforms, batch.src_pixel_values.clone()).cpu())
    orig_tgt_image = Im(undo_normalization_given_transforms(self.cfg.dataset.val.augmentation.tgt_transforms, batch.tgt_pixel_values.clone()).cpu())

    Im.concat_horizontal(
        Im(onehot_to_color(_orig_src_one_hot[0])).resize(orig_tgt_image.height, orig_tgt_image.width).write_text("Src Before Dropout", size=0.5),
        Im(onehot_to_color(_first_src_one_hot__[0])).resize(orig_tgt_image.height, orig_tgt_image.width).write_text("Src After Dropout", size=0.5),
        orig_src_image.resize(orig_tgt_image.height, orig_tgt_image.width).write_text("Src GT"),
        Im.concat_vertical(initial_prompt_images).write_text("Before TTA"),
        Im.concat_vertical(optimized_images).write_text("After TTA"),
        orig_tgt_image.write_text("Tgt GT"),
    ).save(f"tta_{batch.metadata['name'][0]}")

    return {'images': optimized_images[0]}