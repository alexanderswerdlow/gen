exit()
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
    


#             log_info(f"Generating with batch {i}")
#         orig_input_ids = batch["input_ids"].clone()

#         images = []
#         orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
#         gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(
#             text="GT", relative_font_scale=0.004
#         )

#         batch["input_ids"] = orig_input_ids.clone()
#         prompt_image, input_prompt, prompt_embeds = run_inference_batch(
#             batch=batch,
#             model=model,
#             pipeline=pipeline,
#             visualize_attention_map=cfg.inference.visualize_attention_map,
#             inference_cfg=cfg.inference,
#             num_images_per_prompt=cfg.inference.num_images_per_prompt,
#         )

#         full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
#         images.append(
#             Im.concat_horizontal(
#                 Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}", relative_font_scale=0.004)
#                 for i, prompt_image_ in enumerate(prompt_image)
#             )
#         )

#         if cfg.model.break_a_scene_cross_attn_loss:
#             batch_size = (2 if cfg.inference.guidance_scale > 1.0 else 1) * cfg.inference.num_images_per_prompt

#             agg_attn_cond = aggregate_attention(controller=model.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size)
#             attn_img_cond = Im(save_cross_attention_vis(tokenizer=model.tokenizer, tokens=batch["input_ids"][0], attention_maps=agg_attn_cond)).write_text("Cond")

#             agg_attn_uncond = aggregate_attention(controller=model.controller, res=16, from_where=("up", "down"), is_cross=True, select=0, batch_size=batch_size)
#             uncond_tokens = pipeline.tokenizer(
#                 [""],
#                 padding="max_length",
#                 max_length=pipeline.tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             attn_img_uncond = Im(save_cross_attention_vis(tokenizer=model.tokenizer, tokens=uncond_tokens['input_ids'][0], attention_maps=agg_attn_uncond)).write_text("Uncond")

#             all_output_attn_viz.append(Im.concat_vertical(attn_img_cond, attn_img_uncond))

#         elif cfg.inference.visualize_attention_map:
#             desired_res = (64, 64)
#             # inference_cfg.inference.guidance_scale > 1.0
#             attn_maps_per_timestep = retrieve_attn_maps_per_timestep(
#                 image_size=prompt_image.size, timesteps=pipeline.scheduler.timesteps.shape[0], chunk=False
#             )
#             if attn_maps_per_timestep[0].shape[-2] != desired_res[0] or attn_maps_per_timestep[0].shape[-1] != desired_res[1]:
#                 attn_maps_per_timestep = resize_net_attn_map(attn_maps_per_timestep, desired_res)
#             tokens = [x.replace("</w>", "") for x in input_prompt[0]]
#             attn_maps_img_by_timestep = get_all_net_attn_maps(attn_maps_per_timestep, tokens)
#             mask_idx = 0
#             output_cols = []

#             # fmt: off
#             attn_viz_ = Im.concat_vertical(list(
#                     Im.concat_horizontal(attn_maps, spacing=5).write_text(f"Timestep: {pipeline.scheduler.timesteps[idx].item()}", relative_font_scale=0.004)
#                     for idx, attn_maps in enumerate(attn_maps_img_by_timestep)
#             )[::5],spacing=5)
#             # fmt: on

#             for _, (attn_map, token) in enumerate(zip(attn_maps_img_by_timestep[0], tokens)):
#                 if token == model.cfg.inference.model.placeholder_token:
#                     mask_bool = batch["gen_segmentation"][..., mask_idx].squeeze(0).cpu().bool().numpy()
#                     orig_image_ = orig_image.np.copy()
#                     orig_image_[~mask_bool] = 0
#                     orig_image_ = Im(orig_image_, channel_range=ChannelRange.UINT8)
#                     text_to_write = f"mask: {mask_idx}"
#                     mask_idx += 1
#                 else:
#                     orig_image_ = Im(255 * np.ones((desired_res[0], desired_res[1], 3), dtype=np.uint8))
#                     text_to_write = f"{token}"

#                 output_cols.append(orig_image_.resize(*desired_res).write_text(text_to_write, relative_font_scale=0.004))

#             all_output_attn_viz.append(
#                 Im.concat_vertical(attn_viz_, Im.concat_horizontal(output_cols, spacing=5), Im(prompt_image).resize(*desired_res))
#             )

#             if is_main_process() and cfg.inference.visualize_embeds:
#                 embeds_ = torch.stack([v[-1] for k, v in prompt_embeds[0].items() if "CONTEXT_TENSOR" in k and "BYPASS" not in k], dim=0)
#                 bypass_embeds_ = None
#                 if any("BYPASS" in k for k in prompt_embeds[0].keys()):
#                     bypass_embeds_ = torch.stack([v[-1] for k, v in prompt_embeds[0].items() if "CONTEXT_TENSOR" in k and "BYPASS" in k], dim=0)

#                 inputs_ = pipeline.tokenizer(
#                     " ".join([x.replace("</w>", "") for x in input_prompt[0]]),
#                     max_length=pipeline.tokenizer.model_max_length,
#                     padding="max_length",
#                     truncation=True,
#                     return_tensors="pt",
#                 )
#                 inputs_["input_ids"] = inputs_["input_ids"].to(pipeline.text_encoder.device)
#                 inputs_["attention_mask"] = inputs_["attention_mask"].to(pipeline.text_encoder.device)
#                 regular_embeds_ = pipeline.text_encoder(**inputs_).last_hidden_state
#                 embeds_, regular_embeds_ = embeds_[:, : len(input_prompt[0])], regular_embeds_[:, : len(input_prompt[0])]
#                 if bypass_embeds_ is not None:
#                     bypass_embeds_ = bypass_embeds_[:, : len(input_prompt[0])]

#                 try:
#                     output_path_ = output_path / "tmp.png"
#                     embeds_.plt.fig.savefig(output_path_)
#                     log_with_accelerator(
#                         accelerator,
#                         [Im(output_path_)],
#                         global_step=(global_step if global_step is not None else i),
#                         name="embeds",
#                         save_folder=output_path,
#                     )
#                     if bypass_embeds_ is not None:
#                         bypass_embeds_.plt.fig.savefig(output_path_)
#                         log_with_accelerator(
#                             accelerator,
#                             [Im(output_path_)],
#                             global_step=(global_step if global_step is not None else i),
#                             name="bypass_embeds",
#                             save_folder=output_path,
#                         )
#                     regular_embeds_.plt.fig.savefig(output_path_)
#                     log_with_accelerator(
#                         accelerator,
#                         [Im(output_path_)],
#                         global_step=(global_step if global_step is not None else i),
#                         name="regular_embeds",
#                         save_folder=output_path,
#                     )
#                 except:
#                     log_warn("Nan likely found in embeds")

#         if cfg.inference.num_masks_to_remove is not None:
#             gen_segmentation = batch["gen_segmentation"]

#             for j in range(gen_segmentation.shape[-1])[: cfg.inference.num_masks_to_remove]:
#                 log_info(f"Generating with removed mask {j}")
#                 batch["gen_segmentation"] = gen_segmentation[..., torch.arange(gen_segmentation.size(-1)) != j]
#                 batch["input_ids"] = orig_input_ids.clone()
#                 prompt_image, input_prompt, prompt_embeds = run_inference_batch(
#                     batch=batch,
#                     model=model,
#                     pipeline=pipeline,
#                     inference_cfg=cfg.inference,
#                 )
#                 mask_image = Im(get_layered_image_from_binary_mask(gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
#                 mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
#                 mask_alpha = (gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
#                 composited_image = orig_image.pil.copy().convert("RGBA")
#                 composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
#                 composited_image = composited_image.convert("RGB")
#                 images.append(Im.concat_vertical(prompt_image, mask_image, composited_image, spacing=5, fill=(128, 128, 128)))





# def get_image_grid(images: List[Image.Image]) -> Image:
#     num_images = len(images)
#     cols = int(math.ceil(math.sqrt(num_images)))
#     rows = int(math.ceil(num_images / cols))
#     width, height = images[0].size
#     grid_image = Image.new("RGB", (cols * width, rows * height))
#     for i, img in enumerate(images):
#         x = i % cols
#         y = i // cols
#         grid_image.paste(img, (x * width, y * height))
#     return grid_image


# global img_idx
# im_path = f"/home/aswerdlo/repos/gen/output/masks/{img_idx}_mask.png"
# from image_utils import Im
# Im(loss_mask).save(im_path)
# Im((batch["gen_pixel_values"] + 1) / 2).save(im_path.replace("_mask.png", ".png"))
# img_idx += 1
    
            # from image_utils import Im
        # Im(max_masks[-1][..., None]).save(f'loss_{batch["state"].true_step}_{b}.png')
        # Im((batch["gen_pixel_values"][b] + 1) / 2).save(f'img_{batch["state"].true_step}_{b}.png')
    
                # TODO: Something weird happens with webdataset:
            # UserWarning: Length of IterableDataset <abc.WebDataset_Length object at 0x7f0748da4640> was reported to be 2 (when accessing len(dataloader)), but 3 samples have been fetched.
            # if step >= len(self.train_dataloader) - 1:
            #     log_info(f"Exited early at step {global_step}")
            #     break
    
#     torch.sum(feature_map_mask_, dim=[0]) >= 1).sum()

# from image_utils import get_layered_image_from_binary_mask, Im, ChannelRange
# Im(get_layered_image_from_binary_mask(feature_map_mask_.permute(1, 2, 0)), channel_range=ChannelRange.UINT8)



            # attn_proc
            # print(torch.sum(resized_attention_masks, dim=[1, 2]) / (resized_attention_masks.numel() / resized_attention_masks.shape[0]))
            # print(torch.sum(torch.sum(resized_attention_masks, dim=-1) > 1, dim=1) / (latent_dim**2))

            # print(torch.sum(attention_mask[:10], dim=[1, 2]) / (resized_attention_masks.numel() / resized_attention_masks.shape[0]))
            # print("\n\n")
    
            # print(attention_mask.shape, query.shape, key.shape, value.shape)
            # cur_idx = (attn_meta["layer_idx"] - 1) % attn_meta["num_layers"]
            # if cur_idx == 0:
            #     from image_utils import get_layered_image_from_binary_mask, Im, ChannelRange
            #     imgs_ = []
            #     for b in range(batch_size):
            #         img_ = Im(get_layered_image_from_binary_mask(rearrange(attention_mask[b * attn.heads] > -1, '(h w) tokens -> h w tokens', h=latent_dim)), channel_range=ChannelRange.UINT8)
            #         imgs_.append(img_)

            #     global idx_
            #     Im.concat_horizontal(*imgs_).save(f'{idx_}_attn.png')
            #     idx_ += 1

                            # if hasattr(self, "layer_dropout_func"):
                #     input_ = cond.encoder_hidden_states.clone()
                #     self.layer_dropout_func(input_, dropout_idx.nonzero(), self.uncond_hidden_states)
                #     set_at("[b] tokens ([n] d), masked [2], tokens d -> b tokens ([n] d)", cond.encoder_hidden_states, dropout_idx.nonzero(), self.uncond_hidden_states)
                #     assert torch.allclose(input_, cond.encoder_hidden_states)
                # else:
                #     self.layer_dropout_func = set_at("[b] tokens ([n] d), masked [2], tokens d -> b tokens ([n] d)", cond.encoder_hidden_states, dropout_idx.nonzero(), self.uncond_hidden_states, graph=True)

                # set_at("[b] tokens ([n] d), masked [2], tokens d -> b tokens ([n] d)", cond.encoder_hidden_states, dropout_idx.nonzero(), self.uncond_hidden_states)
    
                        # for name, param in self.model.named_parameters():
                        # if param.requires_grad and param.grad is None:
                        #     print(name)
    
                    # def interpolate(step: int, max_steps: int) -> float:
                    # return min(min(max(step, 0) / max_steps, 1.0), 1e-6) # We need to start out >0 so the params are used in the first step
    
            # if "femb" in cond.unet_kwargs:
            # (metadata_dict, clip_feature_maps) = cond.unet_kwargs["femb"]
            # metadata_dict["layer_idx"] = 0

            # base model after dropoutmask
            # if (dropout_mask & non_empty_mask).sum().item() == 0:
            #     assert False, "We should no longer have this condition"
            #     dropout_mask[(~dropout_mask).nonzero(as_tuple=True)[0]] = True
    
                # self.rot_mlp = Mlp(in_features=dim, hidden_features=dim // 4, out_features=6, activation=nn.GELU())
# output = self.rot_mlp(cond.mask_tokens)
    

#     def token_rot_loss(
#     cfg: BaseConfig,
#     batch: InputData,
#     cond: ConditioningData,
#     pred_data: TokenPredData
# ):

#     bs = batch["gen_pixel_values"].shape[0]
#     device = batch["gen_pixel_values"].device

#     assert cfg.model.background_mask_idx == 0

#     losses = []
#     for b in range(bs):
#         if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
#             continue

#         # We align the dataset instance indices with the flattened pred indices
#         mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
#         pred_idxs_for_batch = torch.arange(cond.mask_instance_idx.shape[0], device=device)[cond.mask_batch_idx == b]

#         # The background is always 0 so we must remove it if it exists and move everything else down
#         remove_background_mask = mask_idxs_for_batch != 0

#         pred_idxs_for_batch = pred_idxs_for_batch[remove_background_mask]

#         pred_ = pred_data.pred_6d_rot[pred_idxs_for_batch]
#         gt_rot_6d_ = pred_data.gt_rot_6d[pred_idxs_for_batch]

#         if gt_rot_6d_.shape[0] == 0:
#             continue  # This can happen if we previously dropout all masks except the background

#         loss = F.mse_loss(pred_, gt_rot_6d_)
#         losses.append(loss)

#     return torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=device)

#             scheduler = getattr(self, "scheduler") if hasattr(self, "scheduler") else self.pipeline.scheduler
            # previously we quantized gt for inference metrics to try to make it more fair
            # gt_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d).float().cpu().numpy()).as_quat()
            # gt_discretized_quat = get_quat_from_discretized_zyx(get_discretized_zyx_from_quat(gt_quat, num_bins=num_bins), num_bins=num_bins)
            # pred_data.gt_rot_6d = get_ortho6d_from_rotation_matrix(torch.from_numpy(R.from_quat(gt_discretized_quat).as_matrix()).to(device))
    
        #     camera = rearrange('(b group_size) ... -> b group_size ...', batch["camera_quaternions"].detach().float().cpu().numpy(), group_size=group_size)
        # object = rearrange('(b group_size) ... -> b group_size ...', batch["quaternions"].detach().float().cpu().numpy(), group_size=group_size)
        # # valid = rearrange('(b group_size) ... -> b group_size ...', batch["valid"], group_size=group_size)
        
        # left_cam, right_cam = R.from_quat(camera[:, 0]), R.from_quat(camera[:, 1])
        # camera_delta = (right_cam * left_cam.inv()).as_quat()

        # left_object, right_object = R.from_quat(object[:, 0, 0]), R.from_quat(object[:, 1, 0])
        # obj_delta = (right_object * left_object.inv()).as_quat()
        # assert np.allclose(camera_delta, obj_delta, atol=1e-3)
    
            # if self.relative_val_on_train:
            # # We only ever take the first 12 frames for train
            # assert self.num_frames == 12
            # left, right = zip(*permutation_pairs)
            # permutation_pairs = list(zip([x + self.num_frames for x in left], right)) + list(zip(left, [x + self.num_frames for x in right]))

                            # camera = rearrange('(b group_size) ... -> b group_size ...', batch["camera_quaternions"].detach().float().cpu().numpy(), group_size=group_size)
                # object = rearrange('(b group_size) ... -> b group_size ...', batch["quaternions"].detach().float().cpu().numpy(), group_size=group_size)
                # left_cam, right_cam = R.from_quat(camera[:, 0]), R.from_quat(camera[:, 1])
                # camera_delta = (right_cam * left_cam.inv()).as_quat()
                # left_object, right_object = R.from_quat(object[:, 0, 0]), R.from_quat(object[:, 1, 0])
                # obj_delta = (right_object * left_object.inv()).as_quat()
                # if not np.allclose(camera_delta, obj_delta, atol=1e-3):
                #     log_warn(f"Camera and object rotations are not the same: {camera_delta - obj_delta}")
    
        #     print(self.clip.model.norm.weight, self.clip.model.norm.bias)
        # print(torch.all(self.clip.model.norm.weight == 1), torch.all(self.clip.model.norm.bias == 0))