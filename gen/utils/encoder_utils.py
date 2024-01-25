import functools
import math
from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from typing import Callable, Optional, TypeAlias, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from image_utils import Im
from jaxtyping import Float
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import Tensor
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

from gen.utils.extracted_encoder_utils import (interpolate_embeddings,
                                               pad_image_and_adjust_coords)

ImArr: TypeAlias = Union[Image.Image, Tensor]
import open_clip


def reshape_vit_output(num_to_truncate: int, x: torch.Tensor):
    x = x[:, num_to_truncate:]
    assert np.isclose(int(math.sqrt(x.shape[1])), math.sqrt(x.shape[1]))
    h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
    x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
    return {'x': x}

def identity(**kwargs):
    return kwargs

def get_feature_layer(**kwargs):
    kwargs['x'] = kwargs['x'][-(kwargs['num_from_back'] + 1)]
    return kwargs

def swin_rearrange(**kwargs):
    kwargs = get_feature_layer(**kwargs)
    kwargs['x'] = rearrange(kwargs['x'], 'b h w d -> b d h w')
    return kwargs


class BaseModel(ABC, nn.Module):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        import inspect
        if not inspect.isabstract(cls):
            BaseModel._registry[cls.__name__] = cls

    def __init__(
            self, 
            compile: bool = False, 
            compile_kwargs: dict = {},
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs
        ):
        super().__init__()
        self.model = self.create_model()
        if device is not None:
            self.model = self.model.to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)
        if compile:
            self.model = torch.compile(self.model, **compile_kwargs)

    def create_model_timm(self, model_name: str, **kwargs):
        return timm.create_model(model_name, pretrained=True, **kwargs)

    @abstractmethod
    def create_model(self, **kwargs): pass

    def pre_transform(self, image: ImArr, **kwargs): 
        return image
    
    def post_transform(self, image: ImArr, **kwargs):
        if isinstance(image, Tensor) and image.ndim == 3:
            image = image.unsqueeze(0)
        if isinstance(image, Tensor) and image.device == torch.device('cpu'):
            image = image.to(next(iter(self.model.parameters())).device)
        return image

    @abstractproperty
    def transform(self, *args): return args

    def reshape_func(self, output_data, **kwargs): return output_data

    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs):
        return self.model(image, **kwargs)

    def validate_input(self, image: ImArr, **kwargs):
        pass

    @torch.no_grad()
    def forward(self, image: ImArr, **kwargs):
        input_data = self.pre_transform(image, **kwargs)
        input_data = self.transform(input_data, **kwargs)
        input_data = self.post_transform(input_data, **kwargs)
        output_data = self.forward_model(input_data, **kwargs)
        return self.reshape_func(output_data, **kwargs)
    

class TimmModel(BaseModel):
    def __init__(self, num_from_back: int = 0, tensor_input: bool = False, **kwargs): 
        self.num_from_back = num_from_back
        self.tensor_input = tensor_input
        super().__init__(**kwargs)
    
    @functools.cached_property
    def transform(self): 
        cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        if hasattr(self, 'img_size'):
            cfg['input_size'] = self.img_size
        transform_ = create_transform(**cfg)
        if self.tensor_input:
            transform_.transforms = [x for x in transform_.transforms if not isinstance(x, torchvision.transforms.ToTensor)]
        return transform_

class DINOV2(TimmModel):
    def __init__(self, img_size=(224, 224), **kwargs): 
        self.img_size = img_size
        super().__init__(**kwargs)
    def create_model(self): return super().create_model_timm('vit_base_patch14_reg4_dinov2', img_size=self.img_size)
    def pre_transform(self, image: ImArr, **kwargs): return pad_image_and_adjust_coords(image, patch_size=14)
    def reshape_func(self, output): return reshape_vit_output(x=output, num_to_truncate=5)['x']
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward_features(image, **kwargs)

class VIT(TimmModel):
    def __init__(self, img_size=(224, 224)): 
        self.img_size = img_size
        super().__init__()
    def create_model(self): return self.create_model_timm('vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', img_size=self.img_size)
    def pre_transform(self, image: ImArr, **kwargs): return pad_image_and_adjust_coords(image, patch_size=16)
    def reshape_func(self, output): return reshape_vit_output(x=output, num_to_truncate=1)['x']
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward_features(image)

class ConvNextV2(TimmModel):
    def create_model(self): return self.create_model_timm('convnextv2_base.fcmae_ft_in22k_in1k', features_only=True)
    def pre_transform(self, image: ImArr, **kwargs): return image
    def reshape_func(self, output: Tensor): return get_feature_layer(x=output, num_from_back=self.num_from_back)['x']
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward(image)
    
class SwinV2(TimmModel):
    def create_model(self): return self.create_model_timm('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', features_only=True)
    def pre_transform(self, image: ImArr): return image
    def reshape_func(self, output: Tensor): return swin_rearrange(x=output, num_from_back=self.num_from_back)['x']
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward(image)

class ResNet50(TimmModel):
    def create_model(self): return self.create_model_timm('resnet50.fb_ssl_yfcc100m_ft_in1k', features_only=True)
    def pre_transform(self, image: ImArr, **kwargs): return image
    def reshape_func(self, output: torch.Tensor): return get_feature_layer(x=output, num_from_back=self.num_from_back)['x']
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward(image)

class TorchVisionModel(BaseModel):
    def __init__(self, model_builder: Callable, weights, **kwargs):
        self.model_builder = model_builder
        self.weights = weights
        super().__init__(**kwargs)

    def create_model(self): return self.model_builder(weights=self.weights)
    def pre_transform(self, image: ImArr, **kwargs): return self.transform(image)
    def reshape_func(self, output: torch.Tensor): return output
    def forward_model(self, image: Float[Tensor, "b h w c"], **kwargs): return self.model.forward(image)

class VIT_H_14(TorchVisionModel):
    def __init__(self, img_size=224, **kwargs): 
        self.img_size = img_size
        super().__init__(torchvision.models.vit_h_14, torchvision.models.ViT_H_14_Weights.DEFAULT, **kwargs)

    @functools.cached_property
    def transform(self): return partial(self.weights.transforms, crop_size=self.img_size, resize_size=self.img_size)()

    def create_model(self):
        model = self.model_builder(weights=self.weights)
        new_model_state = interpolate_embeddings(
            image_size=self.img_size,
            patch_size=14,
            model_state=model.state_dict(),
            interpolation_mode="bicubic",  # Default interpolation mode
            reset_heads=False              # Whether to reset the heads or not
        )
        model = self.model_builder(image_size=self.img_size)
        model.load_state_dict(new_model_state)
        return model

class ResNet18TorchVision(TorchVisionModel):
    def __init__(self, img_size=224, **kwargs): 
        self.img_size = img_size
        super().__init__(torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT, **kwargs)

    @functools.cached_property
    def transform(self): return partial(self.weights.transforms, crop_size=self.img_size, resize_size=self.img_size)()

class FeatureExtractorModel(BaseModel):
    def create_model(self, base_model):
        self.base_model = base_model
        return create_feature_extractor(self.base_model, return_nodes=self.return_nodes)
    
    def get_nodes(self):
        return get_graph_node_names(self.base_model)
    
class ClipFeatureExtractor(FeatureExtractorModel):
    def __init__(
            self, 
            model_name: str = 'ViT-L-14', 
            weights: str = 'datacomp_xl_s13b_b90k',
            return_nodes = {
                'transformer.resblocks.0': 'stage0',
                'transformer.resblocks.5': 'stage5',
                'transformer.resblocks.11': 'stage1',
                'transformer.resblocks.17': 'stage17',
                'transformer.resblocks.23': 'stage23',
                'transformer': 'transformer',
                'ln_post': 'ln_post',
            },
            return_only: Optional[str] = None,
            **kwargs
        ):
        self.model_name = model_name
        self.weights = weights
        self.return_nodes = return_nodes
        self.return_only = return_only
        super().__init__(**kwargs)
        
    def create_model(self): 
        base_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name, pretrained=self.weights)
        self.base_model = base_model.visual
        model = super().create_model(self.base_model)
        return model
    
    @functools.cached_property
    def transform(self): return self.preprocess_val

    def forward(self, image: ImArr, **kwargs):
        output = self.model(image)
        if self.return_only is not None:
            output = output[self.return_only]
        return output
    
    def forward_base_model(self, image: ImArr, **kwargs):
        return self.base_model(image)
    
def get_pil_img():
    return Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').scale(0.5).resize(224, 224).pil

def run_all():
    import time
    device = torch.device('cuda:0')
    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').scale(0.5).resize(224, 224).pil
    
    for model_name, model_cls in BaseModel._registry.items():
        model = model_cls()
        model.to(device)
        torch.cuda.synchronize()
        start = time.time()
        output = model(image)
        end = time.time()
        torch.cuda.synchronize()
        print(f'{model_name}: {(end - start) * 1000}ms, {output.shape}')

def simple_example():
    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').scale(0.5).resize(224, 224).pil
    model = ClipFeatureExtractor().cuda()
    output = model(image)
    print(output.keys())

if __name__ == '__main__':
    pass