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
