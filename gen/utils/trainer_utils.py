from gen.configs.base import BaseConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
import os
import shutil

logger = get_logger(__name__)

def handle_checkpointing(cfg: BaseConfig, accelerator: Accelerator, global_step: int):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if cfg.trainer.checkpoints_total_limit is not None:
        checkpoints = os.listdir(cfg.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= cfg.trainer.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - cfg.trainer.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")