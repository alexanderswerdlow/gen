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
