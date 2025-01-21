# Interpertable Image Editing

This repo contains code for an interpertable image editing model. Specifically, we fine-tune a pre-trained text-to-image model to condition on a set of latent features instead of a text prompt. In this way, the diffusion U-Net acts as a decoder and, by training the decoder with augmented data, the latent bottleneck not only learns a useful representation, but allows for precise control over the image.

Depending on the available training data, the architecture varies but broadly, we use augmentation and have the source image (which is encoded into region features), differ from the target image (on which we apply denoising loss). Given an unpaired 2D dataset, this allows for positional control, we we can augment the source image (e.g., flip) and condition the decoder on the region features and the flipped positional embeddings. We also apply random masking given the segmentation (from SAM) and apply a feature similarity loss so that each region feature is disentangled (e.g., the encoding of a car should not depend on a nearby tree). In some settings (e.g., rotation prediction), we have a class and/or rotation prediction head on top of region features to improve their quality, but the pipeline can function self-supervised as well.

This work was inspired by [SODA](https://arxiv.org/abs/2311.17901) which features a similar image encoder/diffusion decoder architecture, but lacks precise control over regions and only handles single objects for 3D editing. The aim of this project was to disentangle indiviudal objects and their attributes.

This is similar to concurrent work, specifically [Neural Assets](https://arxiv.org/abs/2406.09292) which focuses on 3D control and [Editable Image Elements](https://arxiv.org/abs/2404.16029) which performs clustering to obtain a grid of image features to control.

## Overview

We show a high-level architecture below, but we note that this only covers a small set of experiments in this codebase.

![High-Level Architecture](./docs/architecture.png)

### Visualizations

Below we take an initial frame (top-left) and pick a target object (e.g., the block). Then, we keep all latent features exactly the same, but manipulate the position of the block to demonstrate disentanglement.

![Demo 1](./docs/demo_v0.png)

Below we take two frames in a moving scene and interpolate the feature of the flying bowl to demonstrate that the latent space, even without training for specific positional control, is meaningful.

![Demo 1](./docs/demo_v1.png)

Below we demonstrate the ability to remove various objects/regions and have the decoder infill. We train the model with dropout in various regions so that the model does not require each region to be conditioned. In these scenes, the bottom is the original image and we highlight the regions removed in grey.

<div style="display: flex; justify-content: space-between;">
    <img src="./docs/demo_v2.png" width="32%" />
    <img src="./docs/demo_v3.png" width="32%" />
    <img src="./docs/demo_v4.png" width="32%" />
</div>

## Install Instructions

Clone with `--recurse-submodules`. If you forgot, run `git submodule update --init`.


```
export TORCH_CUDA_ARCH_LIST="8.0;8.6" # only works on Ampere or newer [A100, A6000 ADA, A500, etc.]

conda config --set solver libmamba
conda env remove -n gen
conda create -n gen python=3.10
conda activate gen

export CUDA_HOME='/usr/local/cuda-11' # Adjust to your desired cuda location
export MAX_JOBS=4 # This can be increased given more system RAM

pip install 'torch==2.2.*' 'torchvision==0.17.*' --index-url https://download.pytorch.org/whl/cu118
pip install -e diffusers; pip install -e 'image_utils[ALL]'
pip install pip install ninja wheel packaging
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
python setup.py install # Adjust MAX_JOBS higher if you have more RAM
cd csrc/fused_dense_lib && pip install .

pip install -r requirements.txt
```

## Nightly Install (Required for faster SAM)

```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
pip install ninja wheel packaging; pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git
```

Note: You may need to set `export CUDA_HOME="/usr/local/cuda-11"`


## Test Command

```
mkdir data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png --directory-prefix=data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png --directory-prefix=data

python -m accelerate.commands.launch --num_processes 1 train.py +experiment=demo_exp dataset.train_batch_size=2 'dataset.validation_image=[data/conditioning_image_1.png,data/conditioning_image_2.png]' dataset.validation_prompt="red circle with blue background" trainer.eval_every_n_steps=10 exp=example_exp_name tags='[example_tag_1,example_tag_2]'
```

## Training

Global Step: A single gradient update step over all GPUs.
True Step: A single forward pass over all GPUs

We set the effective batch size to be the same regardless of the number of GPUs by setting `enable_dynamic_grad_accum`. This accumulates gradients over multiple forward passes which is equivalent to a single DDP step. We then scale learning rate accordingly as this is required with HF accelerate: `scale_lr_gpus_grad_accum`.

To perform accumulation without gradient synchronization (until the actual backward), we use the Accelerate [accumulate plugin](https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization). Using no_sync directly might be better in the future.

## SLURM

Ideally, since we are using Hydra, we would use the submitit hydra plugin to launch jobs and support hyperparam sweeps with multirun. However, this plugin pickles the python code and calls it directly, making it difficult to call with wrappers (e.g., torchrun or accelerate launch).

Instead, we have `multirun.py` which generates a sequence of calls (through `os.system()`), each of which are a single training run. `launch_slurm.py` then runs a SLURM job.

## Misc

To update submodules, run `git pull --recurse-submodules`

## Profiling 

```
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o $timestamp --force-overwrite true --cudabacktrace=true --osrt-threshold=10000 -x true --capture-range=cudaProfilerApi --stop-on-range-end=true

# To Profile everything, just remove: --capture-range=cudaProfilerApi --stop-on-range-end=true

accelerate launch --main_process_port=$RANDOM --no_python scripts/accelerate_nsys_profile.sh python main.py ...normal args
https://docs.nvidia.com/nsight-systems/UserGuide/index.html#deepspeed
```

## Configs

We use [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) which builds on top of [Hydra](https://hydra.cc/docs/intro/) and [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/).

To modify the config from the command line, refer to this [Hydra guide](https://hydra.cc/docs/advanced/override_grammar/basic/).

Many of the experiments make use of merging global configs to produce a final output. If you want to override the parent config for some module (i.e., set the config to the class defaults and ignore the parents), replace `dict(**kwargs)` with `builds(cls, populate_full_signature=True, zen_partial=True, **kwargs)`.

## Known issues

- Overriding datasets is a bit problematic as all datasets need all kwargs.
- Segmentation maps when cropping sometimes have false where it should be true