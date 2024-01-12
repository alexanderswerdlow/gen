import enum
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import CLIPTextConfig

from gen.models.neti.neti_mapper import NeTIMapper


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    placeholder_token_id: int
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float


class NeTICLIPTextEmbeddings(nn.Module):
    """Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def set_mapper(self, mapper: NeTIMapper):
        self.mapper = mapper

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch: Optional[NeTIBatch] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if batch is not None:
            input_ids = batch.input_ids

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        ####################################################################
        # NeTI logic - Use mapper to overwrite the learnable token embedding
        ####################################################################
        bypass_outputs = None
        if batch is not None:
            mapper_outputs = self.mapper(timestep=batch.timesteps.float(), unet_layer=batch.unet_layers.float(), truncation_idx=batch.truncation_idx)
            mapper_outputs = mapper_outputs.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            if self.mapper.output_bypass:
                bypass_outputs = mapper_outputs[:, mapper_outputs.shape[1] // 2 :]
                mapper_outputs = mapper_outputs[:, : mapper_outputs.shape[1] // 2]

            # Overwrite the index of the placeholder token with the mapper output for each entry in the batch
            # We had to modify this since we now have multiple placeholders per prompt
            learnable_idxs = (input_ids == batch.placeholder_token_id).nonzero(as_tuple=True)
            inputs_embeds[learnable_idxs[0], learnable_idxs[1]] = mapper_outputs[learnable_idxs[0]]
        else:
            bypass_outputs, mapper_outputs = None, None

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings, bypass_outputs, mapper_outputs
