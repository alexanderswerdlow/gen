#!/usr/bin/bash

#export CKPT=outputs/checkpoints/movi_medium_2024-02-03_11_56_43/checkpoint_59000/state/pytorch_model.bin

#CUDA_VISIBLE_DEVICES=0 accelerate launch \
#    --main_process_port=29513 main.py \
#    +experiment=gen \
#    exp=movi_medium \
#    '+modes=[unet_finetune_with_pos_emb,token_pred,movi_medium_two_objects]' \
#    trainer.ckpt=$CKPT \
#    model.freeze_unet=true \
#    trainer.learning_rate=0

if [ ! -d "/scratch/one_object_rotation" ]; then
    cp -rv /projects/katefgroup/aswerdlo/datasets/one_object_rotating/ /scratch/one_object_rotation/
fi
if [ ! -d "/scratch/single_object_rotating" ]; then
    cp -rv /projects/katefgroup/aswerdlo/datasets/single_object_rotating/ /scratch/single_object_rotating/
fi
if [ ! -d "/scratch/multiple_object_rotating" ]; then
    cp -rv /projects/katefgroup/aswerdlo/datasets/multiple_object_rotating/ /scratch/multiple_object_rotating/
fi
if [ ! -d "/scratch/one_object_rotating" ]; then
    cp -rv /projects/katefgroup/aswerdlo/datasets/one_object_rotating/ /scratch/one_object_rotating/
fi
export MOVI_MEDIUM_SINGLE_OBJECT_PATH="/scratch/one_object_rotation/"
export MOVI_MEDIUM_SINGLE_OBJECT_PATH="/scratch/single_object_rotating/"
export MOVI_MEDIUM_SINGLE_OBJECT_PATH="/scratch/multiple_object_rotating/"
export MOVI_MEDIUM_SINGLE_OBJECT_PATH="/scratch/one_object_rotating/"

export PYTHONPATH=$(pwd):$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#     --main_process_port=$RANDOM scripts/visualize_movi_dataset.py \
#     sweep_id=02_20_v2 \
#     +experiment=gen \
#     +modes=[debug_token_pred_discretized,vit_tiny_scratch] \
#     debug=true \
#     model.discretize_rot_bins_per_axis=16 \
#     model.num_conditioning_pairs=8 \
#     model.per_layer_queries=false \
#     model.unet=false \
#     model.freeze_mapper=false \
#     model.freeze_unet=true \
#     model.unet_lora=false \
#     trainer.learning_rate=7e-7 \
#     dataset.train_dataset.num_subset=null

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --main_process_port=$RANDOM main.py \
    sweep_id=02_21_v4 \
    +experiment=gen \
    +modes=[relative_rotation_prediction_v2,disable_relative,vit_small_scratch] \
    debug=true
#     model.num_conditioning_pairs=8 \
#     model.per_layer_queries=false \
#     model.unet=false \
#     model.freeze_mapper=false \
#     model.freeze_unet=true \
#     model.unet_lora=false \
#     trainer.learning_rate=7e-7 \
#     dataset.train_dataset.num_subset=null

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --main_process_port=$RANDOM main.py \
#     sweep_id=02_19_v2 \
#     +experiment=gen \
#     +modes=[debug_token_pred_discretized,vit_tiny_scratch] \
#     debug=true \
#     model.discretize_rot_bins_per_axis=16 \
#     model.num_conditioning_pairs=8 \
#     model.per_layer_queries=false \
#     model.unet=true \
#     model.freeze_mapper=false \
#     model.freeze_unet=true \
#     model.unet_lora=false \
#     model.finetune_unet_with_different_lrs=false \
#     trainer.max_train_steps=10000 \
#     trainer.learning_rate=7e-7 \
#     dataset.train_dataset.num_subset=null

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --main_process_port=$RANDOM main.py \
#     sweep_id=02_19_v2 \
#     +experiment=gen \
#     +modes=[debug_token_pred_discretized,vit_tiny_scratch] \
#     debug=true \
#     model.discretize_rot_bins_per_axis=16 \
#     model.num_conditioning_pairs=8 \
#     model.per_layer_queries=false \
#     model.unet=true \
#     model.freeze_mapper=false \
#     model.freeze_unet=false \
#     model.unet_lora=false \
#     model.finetune_unet_with_different_lrs=true \
#     trainer.learning_rate=7e-7 \
#     trainer.ckpt=outputs/debug/debug_2024-02-20_01_09_08/checkpoints/checkpoint_9999/state/pytorch_model.bin \
#     dataset.train_dataset.num_subset=null