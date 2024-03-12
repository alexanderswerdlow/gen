This code runs SAM on all of COCO. It was taken from https://github.com/pytorch-labs/segment-anything-fast.

We first run `eval_combo.py`, then `postprocess_custom_masks` do so some filtering.

```
export COCO_DIR="/scratch/aswerdlo/coco"
export CKPT_DIR="/home/aswerdlo/repos/lib/segment-anything-fast/experiments_data"
export CACHE_DIR="/scratch/aswerdlow/.cache/sam"

CUDA_VISIBLE_DEVICES=2 python experiments/eval_combo.py --coco_root_dir $COCO_DIR --coco_slice_name train2017 --sam_checkpoint_base_path $CKPT_DIR --sam_model_type vit_h --point_sampling_cache_dir $CACHE_DIR --mask_debug_out_dir $CKPT_DIR/tmp/sam_eval_masks_out --use_local_sam_fork true --use_half bfloat16 --use_compile max-autotune --use_nested_tensor true --batch_size=4 --num_workers 10 --pad_input_image_batch false --split_size 4 --split_idx 0

CUDA_VISIBLE_DEVICES=4 python experiments/eval_combo.py --coco_root_dir $COCO_DIR --coco_slice_name train2017 --sam_checkpoint_base_path $CKPT_DIR --sam_model_type vit_h --point_sampling_cache_dir $CACHE_DIR --mask_debug_out_dir $CKPT_DIR/tmp/sam_eval_masks_out --use_local_sam_fork true --use_half bfloat16 --use_compile max-autotune --use_nested_tensor true --batch_size=4 --num_workers 10 --pad_input_image_batch false --split_size 4 --split_idx 1

CUDA_VISIBLE_DEVICES=5 python experiments/eval_combo.py --coco_root_dir $COCO_DIR --coco_slice_name train2017 --sam_checkpoint_base_path $CKPT_DIR --sam_model_type vit_h --point_sampling_cache_dir $CACHE_DIR --mask_debug_out_dir $CKPT_DIR/tmp/sam_eval_masks_out --use_local_sam_fork true --use_half bfloat16 --use_compile max-autotune --use_nested_tensor true --batch_size=4 --num_workers 10 --pad_input_image_batch false --split_size 4 --split_idx 2

CUDA_VISIBLE_DEVICES=1 python experiments/eval_combo.py --coco_root_dir $COCO_DIR --coco_slice_name train2017 --sam_checkpoint_base_path $CKPT_DIR --sam_model_type vit_h --point_sampling_cache_dir $CKPT_DIR/tmp/sam_coco_mask_center_cache --mask_debug_out_dir $CKPT_DIR/tmp/sam_eval_masks_out --use_local_sam_fork true --use_half bfloat16 --use_compile max-autotune --use_nested_tensor true --batch_size=2 --num_workers 10 --pad_input_image_batch false --split_size 4 --split_idx 3
```

```
python gen/datasets/coco/sam/postprocess_custom_masks.py /scratch/aswerdlo/coco/annotations --splits='val2017'
```