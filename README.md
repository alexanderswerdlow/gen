

## Install Instructions

```
conda create -n gen python=3.10

conda activate gen

pip install 'torch==2.1.*' 'torchvision==0.16.*' 'xformers==0.0.23' --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Nightly Install (Required for faster SAM)

```
pip install https://download.pytorch.org/whl/nightly/cu121/torch-2.3.0.dev20231219%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/nightly/cu121/torchvision-0.18.0.dev20231219%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install -r requirements.txt
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git
```


## Test Command

```
mkdir data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png --directory-prefix=data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png --directory-prefix=data

python -m accelerate.commands.launch --num_processes 1 train.py +experiment=demo_exp dataset.train_batch_size=2 'dataset.validation_image=[data/conditioning_image_1.png,data/conditioning_image_2.png]' dataset.validation_prompt="red circle with blue background" trainer.validation_steps=10 exp=example_exp_name tags='[example_tag_1,example_tag_2]'
```

## WIP

```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name dataset=coco_captions
```