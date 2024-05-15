import antigravity

# if self.cfg.model.duplicate_unet_input_channels and False:
#     pred_xyz, pred_mask = decode_xyz(self.cfg, model_pred, cond.xyz_valid, self.vae, cond.xyz_normalizer)
#     losses['metric_l2_scale_shift_inv'] = get_dustr_loss(batch, pred_xyz, pred_mask)

#     def get_norm_pts(_gt, _pred, _mask):
#         _mask = rearrange('b h w -> (b h w)', _mask)
#         _pred = rearrange('b h w xyz -> (b h w) xyz', _pred)
#         _gt = rearrange('b h w xyz -> (b h w) xyz', _gt)
#         gt_min, gt_max = _gt.min(dim=0)[0], _gt.max(dim=0)[0]
#         _pred = (_pred - gt_min) / (gt_max - gt_min)
#         _gt = (_gt - gt_min) / (gt_max - gt_min)
#         return _gt[_mask], _pred[_mask]

#     losses['metric_valid_norm_xyz_mse'] = F.mse_loss(*get_norm_pts(cond.gt_xyz, pred_xyz, pred_mask), reduction="mean")

        # TODO: You must specify device/dtype
        # cfg.trainer.device = device
        # assert cfg.trainer.dtype == dtype
        # model.to(device=cfg.trainer.device, dtype=cfg.trainer.dtype)
        # if attn_meta.inference_shuffle_per_layer:
        #                     indices = torch.randperm(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)[:trained_views]
        #                     encoder_hidden_states = encoder_hidden_states[:, indices]

                        # elif False:
                        #     latent_model_input = torch.cat([latent_model_input[:1], latent_model_input[1:][torch.randperm(latent_model_input.shape[0]-1, device=latent_model_input.device)]], dim=0)

                        #     other = latent_model_input[1:][torch.randperm(latent_model_input.shape[0]-1, device=latent_model_input.device)]
                        #     n = (other.shape[0] // (training_views - 1)) + 1

                        #     num_pad = (training_views - 1) - (other.shape[0] % (training_views - 1))
                        #     other = torch.cat([other, other[:num_pad]], dim=0)
                            
                        #     other = rearrange('(inference_views ... -> inference_views ...', latent_model_input)
                        #     latent_model_input = torch.cat([latent_model_input[:1], other], dim=0)

                        #     new_latents = torch.zeros_like(latent_model_input)
                          # intrinsics:
  #   name: softmin
  #   num_procrustes_points: 16384
  #   min_focal_length: 0.2
  #   max_focal_length: 3.5
  #   num_candidates: 256
  #   regression:
  #     after_step: 1000
  #     window: 100
