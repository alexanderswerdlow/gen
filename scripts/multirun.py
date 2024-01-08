import subprocess
from itertools import product
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

typer.main.get_command_name = lambda name: name

@app.command()
def main(
    cmd_str: str = '',
    modes: str = '',
    init_str: str = f'''cd ~/aswerdlo/tta && source /grogu/user/mprabhud/miniconda3/bin/activate aswerdlo && export WANDB_API_KEY=ff8ac5cca8f65cb77b0e92087205b9bde874a33c'''
):
    adapt_only_classifier = [False] # [True, False]
    adapt_top_k = [5] # [5, 1000]
    online_mode = ['offline'] # ['online', 'offline']
    use_checkpointing = [True] # [True, False]
    imagenet_c_modes = ['fog'] # , 'gaussian_noise', 'pixelate', 'contrast']
    output_file = "command_output.log"
    # datasets = ['ImageNetCDataset'] # , 'ImageNetADataset'], 'ImageNetDataset']
    datasets = ['ImageNetADataset', 'ImageNetDataset']

    # Generate and run all combinations
    for cls_adapt, top_k, online, checkpointing, dataset, imagenet_c_mode in product(adapt_only_classifier, adapt_top_k, online_mode, use_checkpointing, datasets, imagenet_c_modes):
        name_suffix = f""
        if len(adapt_only_classifier) > 1: name_suffix += f"cls_adapt_{cls_adapt}_"
        if len(adapt_top_k) > 1: name_suffix += f"topk_{top_k}_"
        if len(online_mode) > 1: name_suffix += f"online_{online}_"
        if len(use_checkpointing) > 1: name_suffix += f"checkpointing_{checkpointing}_"
        if len(imagenet_c_modes) > 1: name_suffix += f"imagenet_c_mode_{imagenet_c_mode}_"

        command = f'''{init_str} && \
python launch_slurm.py use_accelerate=False program=main.py +slurm=grogu 'slurm.cmd="\
experiment=dit modes='[base,grogu,{online}{modes}]' exp_suffix='{name_suffix}' \
model.adapt_only_classifier={cls_adapt} tta.adapt_topk={top_k} model.use_checkpointing={checkpointing} input.dataset_name={dataset} input.imagenet_c_mode={imagenet_c_mode} {cmd_str}"' '''
        print(command)
        # with open('filename.txt', 'a') as f: f.write(f'{command}\n')
        # continue
        # Open the output file in append mode and execute the command
        with open(output_file, "a") as file:
            process = subprocess.Popen(command, shell=True, stdout=file, stderr=subprocess.STDOUT)

    # Note: Processes are started and will run concurrently. Their outputs will be appended to the output file.

if __name__ == "__main__":
    app()
