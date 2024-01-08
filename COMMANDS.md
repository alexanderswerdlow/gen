
## WIP

```
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +mode=overfit trainer.log_gradients=10

# Custom
python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name +mode=overfit trainer.log_gradients=10 trainer.eval_every_n_epochs=100 trainer.num_train_epochs=1000

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=controlnet +mode=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.random_subset=8 dataset.train_dataset.conditioning_image_column=image dataset.train_dataset.dataset_name=poloclub/diffusiondb dataset.train_dataset.dataset_config_name=2m_random_1k trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 dataset.train_dataset.caption_column=prompt trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 1 main.py dataset=diffusiondb +experiment=demo_exp exp=example_exp_name trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 1 main.py dataset=diffusiondb +experiment=demo_exp exp=example_exp_name trainer.learning_rate=5e-4 dataset.train_dataset.random_subset=64 dataset.validation_dataset.random_subset=4 trainer.eval_every_n_epochs=2

python -m accelerate.commands.launch --num_processes 4 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e +mode=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.random_subset=8 trainer.eval_every_n_steps=16 trainer.learning_rate=5e-4

python main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=controlnet +mode=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.random_subset=8 dataset.train_dataset.conditioning_image_column=image dataset.train_dataset.dataset_name=poloclub/diffusiondb dataset.train_dataset.dataset_config_name=2m_random_1k trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 dataset.train_dataset.caption_column=prompt trainer.learning_rate=5e-4


python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name trainer.log_gradients=10 dataset=movi_e +mode=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.random_subset=8 trainer.eval_every_n_epochs=200 trainer.num_train_epochs=5000 trainer.learning_rate=5e-4

python -m accelerate.commands.launch --num_processes 1 main.py +experiment=demo_exp exp=example_exp_name dataset=movi_e +mode=overfit dataset.train_dataset.batch_size=8 dataset.train_dataset.random_subset=512 trainer.eval_every_n_steps=250 trainer.eval_every_n_epochs=null trainer.learning_rate=5e-4
```