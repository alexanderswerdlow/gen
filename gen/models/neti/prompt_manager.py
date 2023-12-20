from typing import Optional, List, Dict, Any

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from gen.models.neti.net_clip_text_embedding import NeTIBatch
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel

UNET_LAYERS = ['IN01', 'IN02', 'IN04', 'IN05', 'IN07', 'IN08', 'MID',
               'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11']

SD_INFERENCE_TIMESTEPS = [999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659,
                          639, 619, 599, 579, 559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300,
                          280, 260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20]

class PromptManager:
    """ Class for computing all time and space embeddings for a given prompt. """
    def __init__(
            self, 
            tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel,
            timesteps: List[int] = SD_INFERENCE_TIMESTEPS,
            unet_layers: List[str] = UNET_LAYERS,
            placeholder_token_id: Optional[List] = None,
            placeholder_token: Optional[List] = None,
            torch_dtype: torch.dtype = torch.float32
        ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.unet_layers = unet_layers
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = placeholder_token_id
        self.dtype = torch_dtype

    def embed_prompt(
            self, 
            text: str,
            truncation_idx: Optional[int] = None,
            num_images_per_prompt: int = 1
        ) -> List[Dict[str, Any]]:
        """
        Compute the conditioning vectors for the given prompt. We assume that the prompt is defined using `{}`
        for indicating where to place the placeholder token string. See constants.VALIDATION_PROMPTS for examples.
        """
        text = text.format(self.placeholder_token)
        ids = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # Compute embeddings for each timestep and each U-Net layer
        print(f"Computing embeddings over {len(self.timesteps)} timesteps and {len(self.unet_layers)} U-Net layers.")
        hidden_states_per_timestep = []
        for timestep in tqdm(self.timesteps):
            _hs = {"this_idx": 0}.copy()
            for layer_idx, unet_layer in enumerate(self.unet_layers):
                batch = NeTIBatch(input_ids=ids.to(device=self.text_encoder.device),
                                  timesteps=timestep.unsqueeze(0).to(device=self.text_encoder.device),
                                  unet_layers=torch.tensor(layer_idx, device=self.text_encoder.device).unsqueeze(0),
                                  placeholder_token_id=self.placeholder_token_id,
                                  truncation_idx=truncation_idx)
                layer_hs, layer_hs_bypass = self.text_encoder(batch=batch)
                layer_hs = layer_hs[0].to(dtype=self.dtype)
                _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hs.repeat(num_images_per_prompt, 1, 1)
                if layer_hs_bypass is not None:
                    layer_hs_bypass = layer_hs_bypass[0].to(dtype=self.dtype)
                    _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hs_bypass.repeat(num_images_per_prompt, 1, 1)
            hidden_states_per_timestep.append(_hs)
        print("Done.")
        return hidden_states_per_timestep
