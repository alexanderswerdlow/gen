import autoroot
import functools
from typing import Callable, Optional

from torchvision.models.feature_extraction import create_feature_extractor
from gen.models.encoders.encoder import DINOV2, BaseModel, FeatureExtractorModel, ImArr, TimmModel, ViT, ViTFeatureExtractor
import open_clip
import timm
import math
import torch

from gen.utils.decoupled_utils import breakpoint_on_error

class ClipFeatureExtractor(FeatureExtractorModel):
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        weights: str = "datacomp_xl_s13b_b90k",
        return_nodes={
            "transformer.resblocks.0": "stage1",
            "transformer.resblocks.5": "stage6",
            "transformer.resblocks.11": "stage12",
            "transformer.resblocks.17": "stage18",
            "transformer.resblocks.23": "stage24",
            "transformer": "transformer",
            "ln_post": "ln_post",
        },
        return_only: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.weights = weights
        self.return_nodes = return_nodes
        self.return_only = return_only
        super().__init__(**kwargs)

    def create_model(self):
        base_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name, pretrained=self.weights)
        return create_feature_extractor(base_model.visual, return_nodes=self.return_nodes)

    @functools.cached_property
    def transform(self):
        from gen.datasets.utils import get_open_clip_transforms_v2
        return get_open_clip_transforms_v2()

class ViTWithExtraChannelsFeatureExtractor(FeatureExtractorModel, ViT):
    def __init__(
        self,
        return_nodes={
            "blocks": "blocks",
            "norm": "norm",
            "fc_norm": "fc_norm",
        },
        return_only: Optional[str] = None,
        model_name: str = "vit_base_patch14_reg4_dinov2",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        num_total_input_channels: int = 3,
        **kwargs,
    ):
        self.return_only = return_only
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.num_total_input_channels = num_total_input_channels
        super().__init__(model_name=model_name, return_nodes=return_nodes, **kwargs)

    def reshape_func(self, output):
        return output

    def create_model(self):
        create_kwargs = {}
        if self.num_classes is not None:
            create_kwargs["num_classes"] = self.num_classes

        create_kwargs['pretrained'] = self.pretrained

        if self.img_size is not None:
            create_kwargs["img_size"] = self.img_size

        if "pretrained" not in create_kwargs:
            create_kwargs["pretrained"] = True

        self.base_model = timm.create_model(self.model_name, **create_kwargs)

        print(f"Expanding the number of input channels from 3 to {self.num_total_input_channels}")
        
        self.base_model.patch_embed.proj.in_channels = self.num_total_input_channels
        weight_name = 'patch_embed.proj.weight'
        conv_weight = self.base_model.state_dict()[weight_name]
        conv_type = conv_weight.dtype
        conv_weight = conv_weight.float()
        repeat = int(math.ceil(self.num_total_input_channels / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :self.num_total_input_channels, :, :]
        conv_weight[:, 3:, :, :] = 0
        conv_weight = conv_weight.to(conv_type)
        self.base_model.patch_embed.proj.weight = torch.nn.Parameter(conv_weight)

        return create_feature_extractor(self.base_model, return_nodes=self.return_nodes)

class TransformersModel(BaseModel):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    @functools.cached_property
    def transform(self):
        pass

    def create_model(self):
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        breakpoint()
        return image_processor

class IntermediateViT(TimmModel):
    def __init__(self, model_name: str = "vit_base_patch14_reg4_dinov2", img_size=(224, 224), **kwargs):
        self.model_name = model_name
        super().__init__(model_name=model_name, img_size=img_size, features_only=False, **kwargs)

    def forward_model(self, image, **kwargs):
        _model = self.model.lora_vit if self.lora is not None else self.model
        return {f'blocks.{i}':v for i,v in enumerate(_model.get_intermediate_layers(x=image, n=24 if 'large' in self.model_name else 12, norm=True))}

if __name__ == "__main__":
    with breakpoint_on_error():
        from image_utils import Im
        # model = ViTWithExtraChannelsFeatureExtractor(num_total_input_channels=5).cuda()
        # model = TransformersModel("google/vit-base-patch16-224-in21k").cuda()
        # model = ViTFeatureExtractor(lora=True).cuda()
        # model = IntermediateViT(lora=dict(r=16, alpha=8)).cuda()

        image = Im("https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png").torch
        output = model(image)
        breakpoint()