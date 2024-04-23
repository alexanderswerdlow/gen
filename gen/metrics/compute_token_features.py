from __future__ import annotations

from typing import TYPE_CHECKING

from tensordict import TensorDict
import torch

from gen.utils.decoupled_utils import get_time_sync
from gen.utils.trainer_utils import TrainingState

if TYPE_CHECKING:
    from gen.models.base.base_model import BaseMapper, InputData

@torch.no_grad()
def compute_token_features(self: BaseMapper, batch: InputData, state: TrainingState):
    with torch.cuda.amp.autocast(): 
        cond = self.get_standard_conditioning_for_inference(batch=batch)
    
    metadata_idx = torch.tensor([int(x) for x in batch.metadata['name']]).to(batch.device)
    ret = {
        f'knn_{batch.metadata["split"][0]}': 
           {'mask_tokens': cond.mask_tokens.to(torch.float16).cpu(), 
            'instance_idx': cond.mask_instance_idx.cpu(), 
            'batch_idx': cond.mask_batch_idx.cpu(), 
            'metadata': metadata_idx[cond.mask_batch_idx].cpu(),
            'categories': batch.categories[cond.mask_batch_idx, cond.mask_instance_idx - 1].cpu(),
        }
    }

    return ret