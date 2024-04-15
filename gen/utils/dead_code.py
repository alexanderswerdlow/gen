"""
This file contains code that is no longer needed but for which there is a non-zero chance we might need to refer to it again.
To spare us the sight of this dead code but also the pain of having to dig through git history, we keep it here.
"""
import antigravity
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
# @torch.no_grad()
# def diffusers_eval(
#     cfg: BaseConfig,
#     accelerator: Accelerator,
#     batch: Dict[str, Any],
#     weight_dtype: torch.dtype,
#     model: BaseMapper,
#     noise_scheduler: nn.Module,
#     vae: nn.Module,
#     n: int,
#     max_batch_size: int,
# ):
    
#     unwrap(model.text_encoder).text_model.embeddings.mapper.eval()
#     batch["tgt_pixel_values"] = torch.clamp(batch["tgt_pixel_values"], -1, 1)
#     latents = vae.encode(batch["tgt_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
#     latents = latents * vae.config.scaling_factor

#     # Split timesteps into smaller batches if n is larger than max_batch_size
#     total_timesteps = torch.linspace(0, noise_scheduler.config.num_train_timesteps - 1, steps=n).long()
#     batched_timesteps = total_timesteps.split(max_batch_size)
        
#     total_loss = 0.0
#     from einops import repeat

#     for timesteps in batched_timesteps:
#         bsz = timesteps.shape[0]
#         repeated_latents = latents.repeat(bsz, 1, 1, 1)[:bsz]
#         batch_ = {}
#         for k in batch.keys():
#             batch_[k] = repeat(batch[k][0], '... -> h ...', h=bsz)

#         noise = torch.randn_like(repeated_latents)
#         noisy_latents = noise_scheduler.add_noise(repeated_latents, noise, timesteps.to(latents.device))

#         match cfg.model.model_type:
#             case ModelType.BASE_MAPPER:
#                 model_pred = model(batch_, noisy_latents, timesteps.to(latents.device), weight_dtype)

#         if noise_scheduler.config.prediction_type == "epsilon":
#             target = noise
#         elif noise_scheduler.config.prediction_type == "v_prediction":
#             target = noise_scheduler.get_velocity(repeated_latents, noise, timesteps.to(latents.device))
#         else:
#             raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

#         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
#         total_loss += loss.item()

#     avg_loss = total_loss / len(batched_timesteps)
#     unwrap(model.text_encoder).text_model.embeddings.mapper.train()
#     unwrap(model.text_encoder).train()
#     return avg_loss

# def viz():
#     from image_utils import Im, calculate_principal_components, onehot_to_color, pca

#     principal_components = calculate_principal_components(clip_feature_map.reshape(-1, clip_feature_map.shape[-1]).float())
#     bs_ = clip_feature_map.shape[1]
#     dim_ = clip_feature_map.shape[2]
#     outmap = (
#         pca(
#             clip_feature_map[1:, ...].float().permute(1, 2, 0).reshape(bs_, dim_, 16, 16).permute(0, 2, 3, 1).reshape(-1, dim_).float(),
#             principal_components=principal_components,
#         )
#         .reshape(bs_, 16, 16, 3)
#         .permute(0, 3, 1, 2)
#     )
#     outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
#     outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
#     outmap = (outmap - outmap_min) / (outmap_max - outmap_min)
#     Im(outmap).save("pca")
#     sam_input = rearrange((((batch["tgt_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), "b c h w -> b h w c")
#     Im(sam_input).save("rgb")
#     Im(onehot_to_color(original.permute(1, 2, 0))).save("masks")

# viz()

        # wandb.define_metric("true_step")
        # wandb.define_metric("loss_per_true_step", step_metric="true_step")
# if is_main_process():
#     wandb.log({"loss_per_true_step": loss.detach().item(), "true_step": true_step,}, step=global_step)
    


#             log_info(f"Generating with batch {i}")
#         orig_input_ids = batch["input_ids"].clone()

#         images = []
#         orig_image = Im((batch["tgt_pixel_values"].squeeze(0) + 1) / 2)
#         gt_info = Im.concat_vertical(orig_image, onehot_to_color(batch["tgt_segmentation"].squeeze(0))).write_text(
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

#         full_seg = Im(onehot_to_color(batch["tgt_segmentation"].squeeze(0)))
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
#                     mask_bool = batch["tgt_segmentation"][..., mask_idx].squeeze(0).cpu().bool().numpy()
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
#             tgt_segmentation = batch["tgt_segmentation"]

#             for j in range(tgt_segmentation.shape[-1])[: cfg.inference.num_masks_to_remove]:
#                 log_info(f"Generating with removed mask {j}")
#                 batch["tgt_segmentation"] = tgt_segmentation[..., torch.arange(tgt_segmentation.size(-1)) != j]
#                 batch["input_ids"] = orig_input_ids.clone()
#                 prompt_image, input_prompt, prompt_embeds = run_inference_batch(
#                     batch=batch,
#                     model=model,
#                     pipeline=pipeline,
#                     inference_cfg=cfg.inference,
#                 )
#                 mask_image = Im(onehot_to_color(tgt_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
#                 mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
#                 mask_alpha = (tgt_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
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
# Im((batch["tgt_pixel_values"] + 1) / 2).save(im_path.replace("_mask.png", ".png"))
# img_idx += 1
    
            # from image_utils import Im
        # Im(max_masks[-1][..., None]).save(f'loss_{batch["state"].true_step}_{b}.png')
        # Im((batch["tgt_pixel_values"][b] + 1) / 2).save(f'img_{batch["state"].true_step}_{b}.png')
    
                # TODO: Something weird happens with webdataset:
            # UserWarning: Length of IterableDataset <abc.WebDataset_Length object at 0x7f0748da4640> was reported to be 2 (when accessing len(dataloader)), but 3 samples have been fetched.
            # if step >= len(self.train_dataloader) - 1:
            #     log_info(f"Exited early at step {global_step}")
            #     break
    
#     torch.sum(feature_map_mask_, dim=[0]) >= 1).sum()

# from image_utils import onehot_to_color, Im, ChannelRange
# Im(onehot_to_color(feature_map_mask_.permute(1, 2, 0)), channel_range=ChannelRange.UINT8)



            # attn_proc
            # print(torch.sum(resized_attention_masks, dim=[1, 2]) / (resized_attention_masks.numel() / resized_attention_masks.shape[0]))
            # print(torch.sum(torch.sum(resized_attention_masks, dim=-1) > 1, dim=1) / (latent_dim**2))

            # print(torch.sum(attention_mask[:10], dim=[1, 2]) / (resized_attention_masks.numel() / resized_attention_masks.shape[0]))
            # print("\n\n")
    
            # print(attention_mask.shape, query.shape, key.shape, value.shape)
            # cur_idx = (attn_meta["layer_idx"] - 1) % attn_meta["num_layers"]
            # if cur_idx == 0:
            #     from image_utils import onehot_to_color, Im, ChannelRange
            #     imgs_ = []
            #     for b in range(batch_size):
            #         img_ = Im(onehot_to_color(rearrange(attention_mask[b * attn.heads] > -1, '(h w) tokens -> h w tokens', h=latent_dim)), channel_range=ChannelRange.UINT8)
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

#     bs = batch["tgt_pixel_values"].shape[0]
#     device = batch["tgt_pixel_values"].device

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
        # device: Optional[torch.device]
    # _extras: dict[str, Any] = field(init=False, repr=False)
    # # Below is a hack
    # def __init__(self, **kwargs):
    #     object.__setattr__(self, '_extras', {})
    #     predefined_fields = {f.name for f in fields(self)}
    #     for key, value in kwargs.items():
    #         if key in predefined_fields:
    #             object.__setattr__(self, key, value)
    #         else:
    #             log_warn(f"Setting extra field {key} to {value}")
    #             self._extras[key] = value

    # def __setattr__(self, key, value):
    #     self.__setitem__(key, value)
    
    # def __setitem__(self, key, value):
    #     if hasattr(self, key):
    #         object.__setattr__(self, key, value)
    #     else:
    #         self._extras[key] = value

    # def __getitem__(self, item):
    #     if item in self._extras:
    #         return self._extras[item]
    #     return getattr(self, item)

    # def __delitem__(self, key):
    #     if key in self._extras:
    #         del self._extras[key]
    #     else:
    #         raise KeyError(f"{key} is not an extra field and cannot be deleted.")
            # We have -1 as invalid so we simply add 1 to all the labels to make it start from 0 and then later remove the 1st channel
        # src_data.image = src_data.image.squeeze(0)
        # src_data.segmentation = torch.nn.functional.one_hot(src_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]
        # tgt_data.image = tgt_data.image.squeeze(0)
        # tgt_data.segmentation = torch.nn.functional.one_hot(tgt_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]
        # valid = (torch.sum(src_data.segmentation, dim=[0, 1]) > (src_data.segmentation.shape[0] * self.object_ignore_threshold)**2)
        # categories = torch.full((valid.shape), fill_value=-1)
        # categories[unique_instance] = unique_semantic - 1
        # categories[~valid] = -1
        # valid = valid[..., 1:]
        # categories = categories[..., 1:]
            # from scipy import ndimage
        # import numpy as np
        # def replace_nan_with_nearest(image):
        #     def nan_replace(patch):
        #         if np.isnan(patch[len(patch) // 2]):  # Check if central pixel is NaN
        #             non_nan_patch = patch[~np.isnan(patch)]  # Consider all non-NaN values in the patch
        #             return np.nanmedian(non_nan_patch) if non_nan_patch.size > 0 else np.nan
        #         else:
        #             return patch[len(patch) // 2]  # Return the central pixel if not NaN
            

        #     for c in range(image.shape[2]):
        #         footprint = np.ones((16, 16))
        #         image[:, :, c] = ndimage.generic_filter(image[:, :, c], np.nanmax, size=(10, 10), mode='nearest')

        #     return image
        
        # data.grid[data.grid < 0.1] = torch.nan
        # data.grid = torch.from_numpy(replace_nan_with_nearest(data.grid[0].permute(1, 2, 0).numpy())).permute(2, 0, 1).unsqueeze(0).float()
        # Im(torch.cat((data.grid[0], data.grid[0].new_zeros((1, *data.grid.shape[2:]))), dim=0)).save('1')
    # with open(original_path, 'rb') as original_file:
    # with gzip.open(compressed_path, 'wb', compresslevel=4) as compressed_file:
    #     shutil.copyfileobj(original_file, compressed_file)

    # def decompress_mask(compressed_mask, num_channels: int):
    # """
    # Decompress mask: (B, H, W, M) uint8 -> (B, H, W, C) bool
    # """
    # B, H, W, M = compressed_mask.shape
    # device = compressed_mask.device
    # decompressed_mask = torch.zeros((B, H, W, C), dtype=torch.bool, device=device)
    # valid_mask = compressed_mask < C
    # valid_channels = compressed_mask[valid_mask]
    # valid_indices = valid_mask.nonzero()
    # decompressed_mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2], valid_channels.long()] = True
    # return decompressed_mask
    # def get_one_hot_channels(seg, indices):
#     """
#     Parameters:
#     - seg: [H, W] tensor with integers
#     - indices: [M] tensor with selected indices for one-hot encoding.
#     - N: Number of classes (int).
    
#     Returns:
#     - [H, W, M] tensor representing the one-hot encoded segmentation map for selected indices.
#     """
#     H, W = seg.shape
#     M = len(indices)
    
#     seg_expanded = seg.unsqueeze(-1).expand(H, W, M)
#     indices_expanded = indices.expand(H, W, M)
#     output = (seg_expanded == indices_expanded)
#     return output

# def one_hot_to_integer(one_hot_mask):
#     values, indices = one_hot_mask.max(dim=-1)
#     return torch.where(values > 0, indices, torch.tensor(-1))

# def integer_to_one_hot(int_tensor, num_classes):
#     mask = (int_tensor >= 0) & (int_tensor < num_classes)
#     int_tensor = torch.where(mask, int_tensor, torch.tensor(0))
#     one_hot = torch.nn.functional.one_hot(int_tensor, num_classes)
#     one_hot = torch.where(mask.unsqueeze(-1), one_hot, False)
#     return one_hot

# import io
# import os
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Optional
# from image_utils import library_ops
# import hdf5plugin
# import hickle as hkl
# from image_utils import Im
# from PIL import Image
# from jaxtyping import Float, jaxtyped, Integer
# from beartype import beartype as typechecker
# from torch import Tensor
# from numpy import ndarray


# def image_to_bytes(image):
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format="PNG")
#     return img_byte_arr.getvalue()


# def bytes_to_image(img_bytes):
#     img_byte_arr = io.BytesIO(img_bytes)
#     return Image.open(img_byte_arr, formats=("PNG",))


# @jaxtyped
# @typechecker
# @dataclass(kw_only=True, unsafe_hash=True)
# class GPTConfig:
#     output_dir: str = "output"
#     num_frames: int = 1
#     img_torch: Optional[Float[Tensor, "c h w"]] = None
#     img_np: Optional[Integer[ndarray, "h w c"]] = None
#     img_pil: Optional[Image.Image] = None
#     img: Optional[Im] = None
#     path: Optional[Path] = None

#     def serialize(self, path: Path):
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         dump_dict = asdict(self)
#         for key in list(dump_dict.keys()):
#             if isinstance(dump_dict[key], Im):
#                 dump_dict[f"{key}__im"] = dump_dict[key].np
#                 del dump_dict[key]
#             elif isinstance(dump_dict[key], Image.Image):
#                 dump_dict[f"{key}__pil"] = image_to_bytes(dump_dict[key])
#                 del dump_dict[key]
#         hkl.dump(dump_dict, str(path), mode="w", **hdf5plugin.Zstd())

#     @staticmethod
#     def deserialize(path: Path):
#         obj = hkl.load(str(path))
#         for key in list(obj.keys()):
#             if key.endswith("__im"):
#                 obj[key.removesuffix("__im")] = Im(obj[key])
#                 del obj[key]
#             elif key.endswith("__pil"):
#                 obj[key.removesuffix("__pil")] = bytes_to_image(obj[key])
#                 del obj[key]

#         print(obj)
#         if isinstance(obj, GPTConfig):
#             return obj
#         else:
#             return GPTConfig(**obj)


# file_path = "test.pkl"
# obj = GPTConfig()
# image = Im("validation.png")

# obj.img = image
# obj.img_pil = image.pil
# obj.img_torch = image.torch
# obj.img_np = image.np
# obj.path = Path("test.pkl")

# print(obj)

# obj.serialize(Path(file_path))

# print(f"Compressed: {os.path.getsize(file_path) * 1e-6:.2f} MB")

# obj = GPTConfig.deserialize(Path(file_path))

# print(obj)

# breakpoint()

    # attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)

        # num_channels = min(self.num_instances, seg.max() + 1) if self.num_instances is not None else seg.max() + 1
        # seg = integer_to_one_hot(seg, num_channels=num_channels + 1, add_background_channel=True)

        # try:
#     import sys
#     import IPython
#     sys.breakpointhook = IPython.embed
# except ImportError as e:
#     pass


# def get_distance_matrix(self, poses: np.ndarray) -> torch.Tensor:
#         n = len(poses)
#         distance_matrix = np.zeros((n, n, 2))
#         for i in tqdm(range(n)):
#             for j in range(n):
#                 rotational_distance = self.get_rotational_distance(poses[i], poses[j])
#                 translational_distance = self.get_translational_distance(
#                     poses[i], poses[j]
#                 )
#                 distance_matrix[i, j] = np.array(
#                     [rotational_distance, translational_distance]
#                 )

#         distance_matrix = np.sqrt(np.sum(distance_matrix**2, axis=2))
#         return torch.from_numpy(distance_matrix)

#     def get_rotational_distance(self, pose_1: np.ndarray, pose_2: np.ndarray) -> float:
#         # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=Using%20quaternions%C2%B6&text=The%20difference%20rotation%20quaternion%20that,quaternion%20r%20%3D%20p%20q%20%E2%88%97%20.
#         # https://math.stackexchange.com/questions/90081/quaternion-distance

#         rotation_1 = Rotation.from_matrix(pose_1[:3, :3]).as_quat()
#         rotation_2 = Rotation.from_matrix(pose_2[:3, :3]).as_quat()
#         return 2 * np.arccos(np.dot(rotation_1, rotation_2))

#     def get_translational_distance(
#         self, pose_1: np.ndarray, pose_2: np.ndarray
#     ) -> float:
#         return np.linalg.norm(pose_1[:3, 3] - pose_2[:3, 3])

    # mode_store(
    #     name="soda_coco",
    #     model=dict(
    #         add_pos_emb=False,
    #         add_grid_to_input_channels=True,
    #         encoder=dict(
    #             img_size=224,
    #             num_total_input_channels=5,
    #         ),
    #         feature_map_keys=(
    #             "norm",
    #         ),
    #         decoder_resolution=256,
    #         encoder_resolution=224,
    #         encoder_latent_dim=14,
    #         decoder_latent_dim=32,
    #         unfreeze_last_n_clip_layers=None,
    #         freeze_clip=False,
    #     ),
    #     dataset=dict(
    #         train=dict(
    #             batch_size=36,
    #             augmentation=dict(
    #                 different_src_tgt_augmentation=True,
    #                 enable_random_resize_crop=True, 
    #                 enable_horizontal_flip=True,
    #                 src_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
    #                 tgt_random_scale_ratio=((0.5, 0.9), (0.8, 1.2)),
    #                 enable_rand_augment=False,
    #                 enable_rotate=True,
    #             )
    #         ),
    #         val=dict(
    #             augmentation=dict(
    #                 different_src_tgt_augmentation=True,
    #                 enable_random_resize_crop=True, 
    #                 enable_horizontal_flip=True,
    #                 src_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
    #                 tgt_random_scale_ratio=((0.5, 0.9), (0.8, 1.2)),
    #                 enable_rand_augment=False,
    #                 enable_rotate=True,
    #             )
    #         ),
    #     ),
    #     trainer=dict(
    #         eval_every_n_steps=500,
    #     ),
    #     hydra_defaults=["coco_recon_only", "sam_coco_masks", {"override /model": "basemapper_vit_extra_channels"}],
    # )
        # if self.depth_map:
        #     depth_map_path = (data_dict["path_target"].replace("rgb", "depth").replace("jpg", "png"))
        #     depth_map = Image.open(depth_map_path)
        #     h, w = depth_map.size
        #     depth_map = torchvision.transforms.CenterCrop(min(h, w))(depth_map) # ensure that the depth image corresponds to the target image
        #     depth_map = torchvision.transforms.ToTensor()(depth_map)
        #     ret["depth_map"] = depth_map.float()
    # def read_depth_map(self, depth_map_path: str) -> torch.Tensor:
    #     depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH) # make sure to read the image as 16 bit
    #     depth_map = depth_map.astype(np.int16) # convert to int16, hacky, but depth shouldn't exceed 32.767 m

    # #     return torch.from_numpy(depth_map).unsqueeze(0).float()
    #     if False:
    #         from image_utils import Im
    #         src_seg_viz = onehot_to_color(integer_to_one_hot(src_seg.to(torch.uint8).permute(0, 2, 3, 1))[0])
    #         tgt_seg_viz = onehot_to_color(integer_to_one_hot(tgt_seg.to(torch.uint8).permute(0, 2, 3, 1))[0])
    #         Im.concat_horizontal(
    #             Im.concat_vertical(src_img, src_seg_viz),
    #             Im.concat_vertical(tgt_img, tgt_seg_viz)
    #         ).save(str(idx))
                        # seg = seg.to(torch.uint8).argmax(dim=0)[None, None].to(torch.uint8)

        # image_paths = [path.split('data/')[-1] for path in self.image_files]
        # with open('files.txt', "w") as file: file.write("\n".join(image_paths))
        # rsync -a --files-from=files.txt /projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/data /scratch/aswerdlo/cache/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/data

        # target_file = torchvision.io.read_file(str(image_path))
        # image = torchvision.io.decode_jpeg(target_file, device=self.device)[None] / 255
        # T = self.get_relative_pose(tgt_pose, src_pose)  # shape [7]
        # scene_extent = self.frame_poses[self.dataset_idx_to_frame_pair[idx], :3, 3].max()
# dists = compute_dist(query_data["mask_tokens"], value_data['mask_tokens'])
        # idx = torch.argmin(dists, dim=1)

    # tree = KDTree(train_tokens)
    # dists, idx = tree.query(val_tokens, workers=-1, k=5)
    
    # faiss_index = faiss.index_factory(train_tokens.shape[1], "IVF16384,Flat")
    # faiss_index.verbose = True

    # index = faiss.extract_index_ivf(faiss_index)
    # clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index.d))
    # index.clustering_index = index
    
    # clustering_index.train(train_tokens)
    # # faiss.write_index(faiss_index, load_trained_faiss_index)
    # dists, idx = faiss.search_with_parameters(index, val_tokens, 5)
    # co.shard_type = 1
 # index = faiss.IndexIVFPQ(cpu_index, train_token_dim, nlist, 8, 8)
# index.add(train_tokens)
# index.train(train_tokens)
            # dataset = client.scatter(dataset, broadcast=True)
            # log_info("Scattered dataset...")
                # from peft import LoraConfig, get_peft_model
                # clip_lora_config = LoraConfig(
                #     r=self.cfg.model.clip_lora_rank,
                #     lora_alpha=self.cfg.model.clip_lora_rank,
                #     init_lora_weights="gaussian",
                #     lora_dropout=0.1,
                #     bias="none",
                #     target_modules=["qkv"],
                # )
                # self.clip.base_model_prefix = 'base_model'
                # self.clip = get_peft_model(self.clip, clip_lora_config)
                # cast_training_params(self.clip, dtype=torch.float32)
    #                 def cartesian_to_spherical(self, xyz: np.ndarray) -> np.ndarray:
    #     # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

    #     # ptsnew = np.hstack((xyz, np.zeros(xyz.shape))) #what is this for?
    #     xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    #     z = np.sqrt(xy + xyz[:, 2] ** 2)
    #     # for elevation angle defined from Z-axis down
    #     theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
    #     # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
    #     # # for elevation angle defined from XY-plane up
    #     azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    #     return np.array([theta, azimuth, z])

    # def get_T(self, target_RT: np.ndarray, cond_RT: np.ndarray) -> torch.Tensor:
    #     # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

    #     R, T = target_RT[:3, :3], target_RT[:3, -1]  # double check this
    #     T_target = -R.T @ T

    #     R, T = cond_RT[:3, :3], cond_RT[:3, -1]  # double check this
    #     T_cond = -R.T @ T

    #     theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
    #     theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
    #         T_target[None, :]
    #     )

    #     d_theta = theta_target - theta_cond
    #     d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    #     d_z = z_target - z_cond

    #     d_T = torch.tensor(
    #         [
    #             d_theta.item(),
    #             math.sin(d_azimuth.item()),
    #             math.cos(d_azimuth.item()),
    #             d_z.item(),
    #         ]
    #     )

    #     return d_T
    
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)    
    # dask.config.set({'temporary_directory': "/scratch/aswerdlo/tmp/dask"})
    # dask.config.set({'distributed.admin.tick.limit': '3600s'})
# bs = len(masks)
#     original = torch.from_numpy(np.array([masks[i]['segmentation'] for i in range(bs)]))
#     output, result = find_true_indices_batched(original, 16, 16)

#     from ipdb import set_trace; set_trace()

#     Im(rearrange(original[:, None].repeat(1, 3, 1, 1) * 1.0, 'b c h w -> b h w c')).save('high_res_mask')
#     Im(rearrange(result[:, None].repeat(1, 3, 1, 1) * 1.0, 'b c h w -> b h w c')).scale(64).save('vit_feature_mask')

#     output = mask_max_pool(rearrange(downscaled, 'h w e -> () (h w) e'), rearrange(result, 'b h w -> () b (h w)'))
#     output_feats = output.values

#     principal_components = calculate_principal_components(embeddings, num_components)
#     pca(output_feats.squeeze(0))

#     show_anns(image, masks)

# print('0', torch.sum(src_seg, dim=(1, 2)))

# src_seg = torch.where(src_seg.sum(dim=0) == 0, 255, src_seg.argmax(dim=0))
# [None][None].float()
# breakpoint()
# breakpoint()
# [None].float()
# print('1', torch.unique(src_seg.int(), return_counts=True, sorted=True)[-1], torch.unique(src_seg.int(), return_counts=True)[-2])

# masks = [torch.from_numpy(coco_decode_rle(mask['segmentation'])) for mask in masks]
# scores = torch.tensor([mask.sum() for mask in masks], dtype=torch.float32)
# seg = torch.stack(perform_nms(masks, scores, iou_threshold=0.7)).to(device)
# seg = one_hot_to_integer(seg.permute(1, 2, 0), num_overlapping_masks, assert_safe=False).permute(2, 0, 1)[None]

# src_seg_path = save_data_path / "seg_v0" / scene_id / f"{src_path.stem}.png"
# tgt_seg_path = save_data_path / "seg_v0" / scene_id / f"{tgt_path.stem}.png"

# src_seg_path = save_data_path / "masks" / scene_id / f"{src_path.stem}.msgpack"
# src_seg = test_postprocess(src_seg_path)
# src_seg = one_hot_to_integer(src_seg.permute(1, 2, 0), self.num_overlapping_masks, assert_safe=False).permute(2, 0, 1)
# values, counts = torch.unique(src_seg.int(), return_counts=True, sorted=True)
# src_seg[torch.isin(src_seg, values[counts < 256])] = 255
# src_seg = src_seg[None].float()
# src_seg[src_seg == 255] = -1

# tgt_seg_path = save_data_path / "masks" / scene_id / f"{tgt_path.stem}.msgpack"
# tgt_seg = test_postprocess(tgt_seg_path)
# tgt_seg = one_hot_to_integer(tgt_seg.permute(1, 2, 0), self.num_overlapping_masks, assert_safe=False).permute(2, 0, 1)
# values, counts = torch.unique(tgt_seg.int(), return_counts=True, sorted=True)
# tgt_seg[torch.isin(tgt_seg, values[counts < 256])] = 255
# tgt_seg = tgt_seg[None].float()
# tgt_seg[tgt_seg == 255] = -1
                        # rles = [x['segmentation'] for x in masks]
                        # _masks = torch.cat([torch.from_numpy(mask_utils.decode(rle)).unsqueeze(0) for rle in rles], dim=0)
                        # bbox_area = [x['bbox'][2] * x['bbox'][3] for x in masks]
                        # real_area = [x['area'] for x in masks]
                        # ratios = [x/y for x,y in zip(real_area, bbox_area)]
                        # def plot_bbox_on_image(image: torch.Tensor, bbox_xywh: torch.Tensor, color: tuple = (0, 255, 0), thickness: int = 2) -> torch.Tensor:
                        #     import cv2
                        #     top_left = (int(bbox_xywh[0]), int(bbox_xywh[1]))
                        #     bottom_right = (int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3]))
                        #     image_with_bbox = cv2.rectangle(image, top_left, bottom_right, color, thickness)
                        #     return torch.tensor(image_with_bbox)
                        # Im(torch.stack([plot_bbox_on_image(Im(x).bool_to_rgb().write_text(f"{y:.2f}").np, z['bbox']) for x,y,z in zip(_masks, ratios, masks)])).grid(padding=20, pad_value=0.5).save(f"{scene_id}_{frame_name}")
    #                     def get_client(num_workers, use_slurm: bool = True, adaptive: bool = False):
    # log_info(f"Starting client with {num_workers} workers...", main_process_only=False)
    # if use_slurm:
    #     from dask_jobqueue import SLURMCluster
    #     hostname = socket.gethostname()
    #     if any(s in hostname for s in ("matrix-0-36", "matrix-3-26", "matrix-3-28", "matrix-1-22")):
    #         inferface = "ib1"
    #     else:
    #         inferface = "ib0"

    #     print(f"Using interface {inferface}")
    #     scheduler_host = os.environ['HOSTNAME'].removesuffix(".eth") + ".ib"
    #     cluster = SLURMCluster(
    #         cores=1,
    #         processes=1,
    #         memory="24GB", 
    #         walltime="72:00:00", 
    #         queue="kate_reserved",
    #         job_extra_directives=['--cpus-per-task=8', "--gres=gpu:1", "--constraint=\'A100|6000ADA|A5500|volta|2080Ti\'"], 
    #         # "--exclude=matrix-0-18,matrix-0-22"
    #         # matrix-0-16,matrix-0-36,matrix-3-26,matrix-3-28,matrix-1-22
    #         nanny=False,
    #         log_directory='/home/aswerdlo/repos/gen/outputs/dask',
    #         local_directory="/home/aswerdlo/repos/gen",
    #         scheduler_options={'interface': inferface, 'dashboard_address': ':8787'},
    #         # scheduler_options={"host": scheduler_host, 'dashboard_address': ':8787'},
    #         # worker_extra_args=['--host ${SLURMD_NODENAME}.ib'],
    #         # limit
    #         # interface=inferface,
    #         # env_extra=['module load anaconda', 'source activate mbircone'],
    #         # job_extra=[system_specific_args],
    #     )

    #     if adaptive:
    #         cluster.adapt(minimum_jobs=min(num_workers, 6), num_workers=num_workers)
    #     else:
    #         cluster.scale(num_workers)
        
    #     print(cluster.job_script())
    # else:
    #     from dask.distributed import LocalCluster
    #     cluster = LocalCluster()
    
    # client = Client(cluster)
    # return client
                    # if self.no_filtering is False:
                #     values, counts = torch.unique(src_seg.int(), return_counts=True, sorted=True)
                #     src_seg[torch.isin(src_seg, values[counts < 128])] = 255
                    #             def find_all_linear_modules(model):
                    # lora_module_names = set()
                    # for name, module in model.named_modules():
                    #     if isinstance(module, (torch.nn.Linear)):
                    #         names = name.split(".")
                    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                    #         if "lm_head" in lora_module_names: # needed for 16-bit
                    #             lora_module_names.remove("lm_head")
                    # return list(lora_module_names)
                #                 elif False and self.no_filtering is False and self.allow_instance_seg is False:
                # max_num_masks = 20
                # _unique = torch.unique(data_.segmentation)
                # _unique = _unique[_unique != 255]
                # _allowed = torch.cat([_unique[:max_num_masks], torch.tensor([255])])
                # data_.segmentation[~torch.isin(data_.segmentation, _allowed)] = 255
                        # idxs = idxs[:int(self.cfg.inference.num_masks_to_remove * 1.2)]
        # idxs = idxs[torch.randperm(len(idxs))]
        #         idxs = idxs[:int(self.cfg.inference.num_masks_to_remove * 1.5)]
        # idxs = idxs[torch.randperm(len(idxs))]

        # batch.one_hot_tgt_segmentation = integer_to_one_hot(batch.tgt_segmentation, num_classes=self.cfg.model.segmentation_map_size)
# image_batch_tokens = {}
# for b in range(batch.bs):
#     orig_image = Im((batch.tgt_pixel_values[b] + 1) / 2)
#     all_masks = []
#     for j in orig_cond.mask_instance_idx[orig_cond.mask_batch_idx == b]:
#         composited_image = get_composited_mask(batch, b, j)
#         all_masks.append(composited_image.write_text(f"token_{j}").resize(self.cfg.model.decoder_resolution, self.cfg.model.decoder_resolution).np)
#     image_batch_tokens[b] = {"mask_tokens": orig_cond.mask_tokens[orig_cond.mask_batch_idx == b], "mask_rgb": np.stack(all_masks), "orig_image": orig_image.np}

            # _output = self.mapper.inject_positional_information_film(cond.mask_token_pos_emb)
            # scale, shift = einops.rearrange(_output, "b (n a) -> a b n", a=2)
            # cond.mask_tokens = cond.mask_tokens * (1 - scale) + shift
            # logging.basicConfig(level=logging.INFO, format= "%(message)s", datefmt="[%X]", handlers=[RichHandler()])

# def get_single_frame(self, index):
#         try:
#             data = self.hypersim.__getitem__(index)
#         except Exception as e:
#             log_error(e)
#             return self.__getitem__(random.randint(0, len(self))) # Very hacky but might save us in case of an error with a single instance.

#         ret = {}

#         if self.return_raw_dataset_image: ret["raw_dataset_image"] = data["rgb"].copy()

#         rgb, seg, metadata = torch.from_numpy(data['rgb']).to(self.device), torch.from_numpy(data['instance']).to(self.device), data["identifier"]
#         rgb = rearrange(rgb / 255, "h w c -> () c h w")
#         seg = rearrange(seg.to(torch.float32), "h w -> () () h w")

#         if len(torch.unique(seg)) <= 1:
#             raise Exception(f"Segmentation mask has only one unique value for index {index}")
        
#         src_data, tgt_data = self.augmentation(
#             src_data=Data(image=rgb, segmentation=seg),
#             tgt_data=Data(image=rgb, segmentation=seg),
#             use_keypoints=False, 
#             return_encoder_normalized_tgt=self.return_encoder_normalized_tgt
#         )

#         if self.return_encoder_normalized_tgt:
#             tgt_data, tgt_data_src_transform = tgt_data

#         def process_data(data_: Data):
#             data_.image = data_.image.squeeze(0)
#             data_.segmentation = rearrange(data_.segmentation, "() c h w -> h w c")
#             data_.segmentation[data_.segmentation == -1] = 255
#             data_.segmentation = torch.cat([data_.segmentation, data_.segmentation.new_full((*data_.segmentation.shape[:-1], self.num_overlapping_masks - 1), 255)], dim=-1)
#             data_.pad_mask = ~(data_.segmentation < 255).any(dim=-1)
#             return data_
        
#         src_data = process_data(src_data)
#         tgt_data = process_data(tgt_data)

#         if self.return_encoder_normalized_tgt:
#             tgt_data_src_transform = process_data(tgt_data_src_transform)
#             ret.update({
#                 "tgt_enc_norm_pixel_values": tgt_data_src_transform.image,
#                 "tgt_enc_norm_segmentation": tgt_data_src_transform.segmentation.to(torch.uint8),
#                 "tgt_enc_norm_valid": torch.full((255,), True, dtype=torch.bool),
#             })

#         pixels = src_data.segmentation.long().contiguous().view(-1)
#         pixels = pixels[(pixels < 255) & (pixels >= 0)]
#         src_bincount = torch.bincount(pixels, minlength=self.top_n_masks_only)
#         valid = src_bincount > 0

#         name = "_".join(metadata)

#         extrinsics = data['extrinsics']
#         rot = R.from_quat((extrinsics['quat_x'], extrinsics['quat_y'], extrinsics['quat_z'], extrinsics['quat_w']))

#         # Normalize translation
#         # camera_trajectory_extent = self.camera_trajectory_extents[(metadata[0], metadata[1])]
#         T = torch.tensor([extrinsics['x'], extrinsics['y'], extrinsics['z']]).view(3, 1) / 50
        
#         RT = torch.cat((torch.from_numpy(rot.as_matrix()), T), dim=1)
#         RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]])), dim=0)

#         ret.update({
#             "tgt_pad_mask": tgt_data.pad_mask,
#             "tgt_pixel_values": tgt_data.image,
#             "tgt_segmentation": tgt_data.segmentation.to(torch.uint8),
#             "src_pad_mask": src_data.pad_mask,
#             "src_pixel_values": src_data.image,
#             "src_segmentation": src_data.segmentation.to(torch.uint8),
#             "src_pose": RT,
#             "tgt_pose": RT,
#             "src_valid": valid,
#             "input_ids": get_tokens(self.tokenizer),
#             "valid": valid[..., 1:],
#             "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
#             "has_global_instance_ids": torch.tensor(True),
#             "metadata": {
#                 "dataset": "hypersim",
#                 "name": name,
#                 "scene_id": metadata[0],
#                 "camera_frame": metadata[2],
#                 "index": index,
#                 "camera_trajectory": metadata[1],
#                 "split": self.split.name.lower(),
#                 "frame_idxs": (0, 0) # Dummy value
#             },
#         })

#         if src_data.grid is not None: ret["src_grid"] = src_data.grid.squeeze(0)
#         if tgt_data.grid is not None: ret["tgt_grid"] = tgt_data.grid.squeeze(0)

#         return ret

            # if self.cfg.model.max_num_training_masks is not None and (self.training or batch.treat_as_train_batch):
            #     keep_idxs = combined_mask.nonzero().squeeze(1)
            #     keep_idxs = keep_idxs[torch.randperm(keep_idxs.shape[0])]
            #     keep_idxs = keep_idxs[:self.cfg.model.max_num_training_masks]
            #     combined_mask = combined_mask.new_full((combined_mask.shape[0],), False, dtype=torch.bool)
            #     combined_mask[keep_idxs] = True
                    #         elif False:
                    # log_warn("We would have dropped all masks but instead we preserved the background", main_process_only=False)
                    # dropout_mask[background_mask_idx] = True