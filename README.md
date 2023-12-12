

## Install Instructions

```
conda create -n gen python=3.10

conda activate gen

pip install 'torch==2.1.*' 'torchvision==0.16.*' 'xformers==0.0.23' --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```


## Test Command

```
mkdir data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png --directory-prefix=data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png --directory-prefix=data

python -m accelerate.commands.launch --num_processes 1 train.py +experiment=demo_exp dataset.train_batch_size=2 'dataset.validation_image=[data/conditioning_image_1.png,data/conditioning_image_2.png]' dataset.validation_prompt="red circle with blue background" trainer.validation_steps=10 exp=example_exp_name tags='[example_tag_1,example_tag_2]'
```