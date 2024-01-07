

## Install Instructions

```
conda create -n gen python=3.10

conda activate gen

pip install 'torch==2.1.*' 'torchvision==0.16.*' 'xformers==0.0.23' --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Nightly Install (Required for faster SAM)

```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
pip install ninja; pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git
```


## Test Command

```
mkdir data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png --directory-prefix=data
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png --directory-prefix=data

python -m accelerate.commands.launch --num_processes 1 train.py +experiment=demo_exp dataset.train_batch_size=2 'dataset.validation_image=[data/conditioning_image_1.png,data/conditioning_image_2.png]' dataset.validation_prompt="red circle with blue background" trainer.num_val_steps=10 exp=example_exp_name tags='[example_tag_1,example_tag_2]'
```

## WIP

```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +mode=overfit trainer.log_gradients=10

# Custom
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +mode=overfit trainer.log_gradients=10 trainer.eval_every_n_epochs=100 trainer.num_train_epochs=1000

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=controlnet +mode=overfit dataset.train_dataset.random_subset=4 dataset.train_dataset.conditioning_image_column=image dataset.train_dataset.dataset_name=poloclub/diffusiondb dataset.train_dataset.dataset_config_name=2m_random_1k trainer.eval_every_n_epochs=100 trainer.num_train_epochs=1000 dataset.train_dataset.caption_column=prompt
```