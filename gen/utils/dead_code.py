# if cfg.model.tmp_revert_to_neti_logic:
#     avg_loss_eval = diffusers_eval(
#         cfg=cfg,
#         accelerator=accelerator,
#         batch=batch,
#         weight_dtype=weight_dtype,
#         model=model,
#         noise_scheduler=noise_scheduler,
#         vae=vae,
#         n=200,
#         max_batch_size=20,
#     )
#     log_info(f"Eval loss: {avg_loss_eval} at step {global_step}")
#     wandb.log({"avg_eval_loss": avg_loss_eval}, step=global_step)
@torch.no_grad()
def diffusers_eval(
    cfg: BaseConfig,
    accelerator: Accelerator,
    batch: Dict[str, Any],
    weight_dtype: torch.dtype,
    model: BaseMapper,
    noise_scheduler: nn.Module,
    vae: nn.Module,
    n: int,
    max_batch_size: int,
):
    
    unwrap(model.text_encoder).text_model.embeddings.mapper.eval()
    batch["gen_pixel_values"] = torch.clamp(batch["gen_pixel_values"], -1, 1)
    latents = vae.encode(batch["gen_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Split timesteps into smaller batches if n is larger than max_batch_size
    total_timesteps = torch.linspace(0, noise_scheduler.config.num_train_timesteps - 1, steps=n).long()
    batched_timesteps = total_timesteps.split(max_batch_size)
        
    total_loss = 0.0
    from einops import repeat

    for timesteps in batched_timesteps:
        bsz = timesteps.shape[0]
        repeated_latents = latents.repeat(bsz, 1, 1, 1)[:bsz]
        batch_ = {}
        for k in batch.keys():
            batch_[k] = repeat(batch[k][0], '... -> h ...', h=bsz)

        noise = torch.randn_like(repeated_latents)
        noisy_latents = noise_scheduler.add_noise(repeated_latents, noise, timesteps.to(latents.device))

        match cfg.model.model_type:
            case ModelType.BASE_MAPPER:
                model_pred = model(batch_, noisy_latents, timesteps.to(latents.device), weight_dtype)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(repeated_latents, noise, timesteps.to(latents.device))
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        total_loss += loss.item()

    avg_loss = total_loss / len(batched_timesteps)
    unwrap(model.text_encoder).text_model.embeddings.mapper.train()
    unwrap(model.text_encoder).train()
    return avg_loss

def viz():
    from image_utils import Im, calculate_principal_components, get_layered_image_from_binary_mask, pca

    principal_components = calculate_principal_components(clip_feature_map.reshape(-1, clip_feature_map.shape[-1]).float())
    bs_ = clip_feature_map.shape[1]
    dim_ = clip_feature_map.shape[2]
    outmap = (
        pca(
            clip_feature_map[1:, ...].float().permute(1, 2, 0).reshape(bs_, dim_, 16, 16).permute(0, 2, 3, 1).reshape(-1, dim_).float(),
            principal_components=principal_components,
        )
        .reshape(bs_, 16, 16, 3)
        .permute(0, 3, 1, 2)
    )
    outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
    outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
    outmap = (outmap - outmap_min) / (outmap_max - outmap_min)
    Im(outmap).save("pca")
    sam_input = rearrange((((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), "b c h w -> b h w c")
    Im(sam_input).save("rgb")
    Im(get_layered_image_from_binary_mask(original.permute(1, 2, 0))).save("masks")

# viz()

        # wandb.define_metric("true_step")
        # wandb.define_metric("loss_per_true_step", step_metric="true_step")
# if is_main_process():
#     wandb.log({"loss_per_true_step": loss.detach().item(), "true_step": true_step,}, step=global_step)