from gc import disable
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import CLIPTokenizer

from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.net_clip_text_embedding import NeTIBatch
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.utils.decoupled_utils import is_main_process
from gen.utils.trainer_utils import custom_ddp_unwrap

UNET_LAYERS = ["IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "MID", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]

SD_INFERENCE_TIMESTEPS = [
    999,
    979,
    959,
    939,
    919,
    899,
    879,
    859,
    839,
    819,
    799,
    779,
    759,
    739,
    719,
    699,
    679,
    659,
    639,
    619,
    599,
    579,
    559,
    539,
    519,
    500,
    480,
    460,
    440,
    420,
    400,
    380,
    360,
    340,
    320,
    300,
    280,
    260,
    240,
    220,
    200,
    180,
    160,
    140,
    120,
    100,
    80,
    60,
    40,
    20,
]


class PromptManager:
    """Class for computing all time and space embeddings for a given prompt."""

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: NeTICLIPTextModel,
        timesteps: List[int] = SD_INFERENCE_TIMESTEPS,
        unet_layers: List[str] = UNET_LAYERS,
        placeholder_token_id: Optional[List] = None,
        placeholder_token: Optional[List] = None,
        torch_dtype: torch.dtype = torch.float32,
        model: Optional[BaseMapper] = None,
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.unet_layers = unet_layers
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = placeholder_token_id
        self.dtype = torch_dtype
        self.model = model

    def embed_prompt(
        self,
        batch: dict,
        truncation_idx: Optional[int] = None,
        num_images_per_prompt: int = 1,
        disable_conditioning: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Compute the conditioning vectors for the given prompt. We assume that the prompt is defined using `{}`
        for indicating where to place the placeholder token string. See constants.VALIDATION_PROMPTS for examples.
        """
        custom_ddp_unwrap(self.model).weight_dtype = self.dtype
        input_ids, text_encoder_dict, input_prompt = custom_ddp_unwrap(self.model).get_hidden_state(
            batch,
            timesteps=self.timesteps,
            device=batch["gen_pixel_values"].device,
            dtype=self.dtype,
            per_timestep=True,
            disable_conditioning=disable_conditioning,
        )

        # Compute embeddings for each timestep and each U-Net layer
        print(f"Computing embeddings over {len(self.timesteps)} timesteps and {len(self.unet_layers)} U-Net layers.")
        hidden_states_per_timestep = []
        for timestep in tqdm(self.timesteps, leave=False, disable=not is_main_process()):
            _hs = {"this_idx": 0}.copy()
            for layer_idx, unet_layer in enumerate(self.unet_layers):
                neti_batch = NeTIBatch(
                    input_ids=input_ids.to(device=self.text_encoder.device),
                    placeholder_token_id=self.placeholder_token_id,
                    timesteps=timestep.unsqueeze(0).to(device=self.text_encoder.device),
                    unet_layers=torch.tensor(layer_idx, device=self.text_encoder.device).unsqueeze(0),
                    truncation_idx=truncation_idx,
                )
                layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=neti_batch, **text_encoder_dict)
                layer_hidden_state = layer_hidden_state[0].to(dtype=self.dtype)
                _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state.repeat(num_images_per_prompt, 1, 1)
                if layer_hidden_state_bypass is not None:
                    layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.dtype)
                    _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass.repeat(num_images_per_prompt, 1, 1)
            hidden_states_per_timestep.append(_hs)
        return hidden_states_per_timestep, input_prompt
