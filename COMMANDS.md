
## WIP

```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +modes=overfit trainer.log_gradients=10

# Custom
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +modes=overfit trainer.log_gradients=10 trainer.eval_every_n_epochs=100 trainer.num_train_epochs=1000

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=controlnet +modes=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.subset_size=8 dataset.train_dataset.conditioning_image_column=image dataset.train_dataset.dataset_name=poloclub/diffusiondb dataset.train_dataset.dataset_config_name=2m_random_1k trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 dataset.train_dataset.caption_column=prompt trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 1 main.py dataset=diffusiondb +experiment=demo_exp exp=example_exp_name trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 1 main.py dataset=diffusiondb +experiment=demo_exp exp=example_exp_name trainer.learning_rate=5e-4 dataset.train_dataset.subset_size=64 dataset.validation_dataset.subset_size=4 trainer.eval_every_n_epochs=2

python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e +modes=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.subset_size=8 trainer.eval_every_n_steps=16 trainer.learning_rate=5e-4

python main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=controlnet +modes=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.subset_size=8 dataset.train_dataset.conditioning_image_column=image dataset.train_dataset.dataset_name=poloclub/diffusiondb dataset.train_dataset.dataset_config_name=2m_random_1k trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 dataset.train_dataset.caption_column=prompt trainer.learning_rate=5e-4


python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=movi_e +modes=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.subset_size=8 trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e dataset.train_dataset.batch_size=11 trainer.eval_every_n_steps=2000 trainer.eval_every_n_epochs=null trainer.learning_rate=5e-5 trainer.checkpointing_steps=5000

python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=overfit_512_exp dataset=movi_e +modes=overfit dataset.train_dataset.batch_size=10 dataset.train_dataset.subset_size=512 trainer.eval_every_n_steps=1000 trainer.eval_every_n_epochs=null trainer.learning_rate=5e-5 trainer.checkpointing_steps=1000

python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=overfit_512_exp dataset=movi_e +modes=overfit dataset.train_dataset.batch_size=10 dataset.train_dataset.subset_size=512 trainer.eval_every_n_steps=500 trainer.eval_every_n_epochs=null trainer.learning_rate=8e-3 trainer.checkpointing_steps=1000


python -m accelerate.commands.launch --num_processes 1 main.py run_inference=true inference.input_dir=outputs/train/example_exp_name_2024-01-08_15-14-50 inference.iteration=30000
```

## Train
```
python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e dataset.train_dataset.batch_size=11 trainer.eval_every_n_steps=2000 trainer.eval_every_n_epochs=null trainer.learning_rate=5e-5 trainer.checkpointing_steps=5000

python -m accelerate.commands.launch --num_processes 6 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e dataset.train_dataset.batch_size=8 trainer.eval_every_n_steps=1000 trainer.eval_every_n_epochs=null trainer.learning_rate=5e-5 trainer.checkpointing_steps=1000 model.unfreeze_last_n_clip_layers=6 trainer.gradient_accumulation_steps=4
```

## Inference

```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=inference dataset=movi_e dataset.train_dataset.batch_size=4 dataset.train_dataset.subset_size=4 trainer.eval_every_n_steps=null trainer.eval_every_n_epochs=50 trainer.learning_rate=5e-4 trainer.num_train_epochs=50

python -m accelerate.commands.launch --num_processes 1 main.py run_inference=true inference.input_dir=outputs/train/inference_2024-01-08_13-55-09 inference.iteration=last
```

## Super quick training
```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=gen exp=inference +modes=fast
```

outputs/train/inference_2024-01-09_18-48-55/checkpoints

python -m accelerate.commands.launch --num_processes 1 main.py run_inference=true inference.input_dir=outputs/train/inference_2024-01-09_18-48-55 inference.iteration=last

## Imagenet

```
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29511 --num_processes 6 main.py +experiment=gen exp=inference '+modes=[overfit_movi,overfit_imagenet]' trainer.eval_on_start=true dataset=imagenet
```