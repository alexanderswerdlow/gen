import autoroot
from einops import rearrange
import torch
import os

device_name = torch.cuda.get_device_name()
if not device_name.startswith('NVIDIA A100'):
    print("Warning: Custom flash attention kernels were written specifically for A100. Setting SEGMENT_ANYTHING_FAST_USE_FLASH_4=0")
    os.environ['SEGMENT_ANYTHING_FAST_USE_FLASH_4'] = '0'

from pathlib import Path
from gen.utils.decoupled_utils import load_checkpoint_from_url
from image_utils import Im
import numpy as np
import torch.nn as nn
import time

from segment_anything_fast import SamAutomaticMaskGenerator, SamPredictor, sam_model_fast_registry as sam_model_registry
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_urls = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

class HQSam(nn.Module):
    def __init__(
            self,
            model_type = "vit_b",
            device="cuda",
            dtype=torch.float32,
        ):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=load_checkpoint_from_url(model_urls[model_type]))
        self.sam = self.sam.to(device=device, dtype=dtype)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def forward(self, image):
        """
        image (np.ndarray): The image to generate masks for, in HWC uint8 format.
        """
        return self.mask_generator.generate(image)

def show_anns(image, anns, output_path: Path = Path("outputs/sam_hq.png")):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def run_timing_exps():
    num_iters = 2
    device = 'cuda'

    hqsam = HQSam(sam_checkpoint='data/sam_hq_vit_tiny.pth', model_type='vit_tiny', device=device, compile=True)
    hqsam = hqsam.to(device)
    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example7.png').scale(0.5).np

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iters):
        masks = hqsam.forward(image)
    torch.cuda.synchronize()
    print("Inference time auto masks: ", (time.time() - start_time) / num_iters)

    predictor = SamPredictor(hqsam.sam)
    predictor.set_image(image)
    hq_token_only = False
    input_box = torch.tensor([[45,260,515,470], [310,228,424,296]], device=device)
    transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
    input_point, input_label = None, None

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iters):
        masks, scores, logits = predictor.predict_torch(
            point_coords=input_point,
            point_labels=input_label,
            boxes = transformed_box,
            multimask_output=False,
            hq_token_only=hq_token_only, 
        )
    torch.cuda.synchronize()
    print("Inference time prompts: ", (time.time() - start_time) / num_iters)

def test_params():
    def dict_to_filename(d):
        return '_'.join(f"{key}={value}" for key, value in d.items()) + '.txt'

    device='cuda'

    sam_checkpoint='data/sam_vit_b_01ec64.pth'
    model_type='vit_b'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    # sam_checkpoint = '/home/aswerdlow/Documents/research/gen/data/sam_hq_vit_tiny.pth'
    # model_type = 'vit_tiny'
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam = sam.to(device=device)

    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').scale(0.5).np
    num_iters = 5
    
    params = (
        dict(points_per_side=8),
        dict(points_per_side=8, process_batch_size=128),
        dict(points_per_side=8, crop_n_layers=1, crop_n_points_downscale_factor=2),
        dict(points_per_side=32, process_batch_size=4),
        dict(points_per_side=32, crop_n_layers=1, crop_n_points_downscale_factor=2,  process_batch_size=4)
    )

    for i in params:
        init_params = dict(
            process_batch_size=8
        )
        init_params.update(i)
        # if 'process_batch_size' in init_params:
        #     del init_params['process_batch_size']
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            **init_params
        )
        for j in range(num_iters):
            if j == 1:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            masks = mask_generator.generate(image)
            # mask_generator.predictor.set_image(image)
            # masks, scores, logits = mask_generator.predictor.predict_torch(
            # point_coords=input_points,
            # point_labels=input_labels,
            # boxes = None,
            # multimask_output=False,
        
        end_event.record()
        torch.cuda.synchronize()
        max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        _, total_memory = torch.cuda.mem_get_info()
        max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
        max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
        filename = dict_to_filename(init_params)
        print(f"{filename}: time: {(start_event.elapsed_time(end_event) / 1000.) / (num_iters - 1)}, memory(MiB): {max_memory_allocated_bytes} memory(%): {max_memory_allocated_percentage}")

        torch.cuda.empty_cache()
        show_anns(image, masks, (Path('outputs') / filename).with_suffix('.png'))
        # from segment_anything_fast import tools
        # tools.apply_eval_dtype_predictor(predictor, use_half)

def find_true_indices_batched(original, dh, dw):
    # Get dimensions
    batch_size, h, w = original.shape
    # dh, dw, d = downscaled.shape
    
    # Reshape and unfold to align with the downscaled dimensions
    reshaped = original.unfold(1, h // dh, h // dh).unfold(2, w // dw, w // dw)
    reshaped = reshaped.reshape(batch_size, dh, dw, -1)

    # Check for any True values in the corresponding blocks
    result = reshaped.any(dim=3)

    # Get indices where result is True
    indices = [torch.nonzero(r, as_tuple=False) for r in result]

    return indices, result

def mask_max_pool(embeddings, mask):
    '''
    Inputs:
    ------------------
    embeddings: [B, D, E], 
    mask: [B, R, D], 0s and 1s, 1 indicates membership

    Outputs:
    ------------------
    max pooled embeddings: [B, R, E], the max pooled embeddings according to the membership in mask
    max pooled index: [B, R, E], the max pooled index
    '''
    B, D, E = embeddings.shape
    _, R, _ = mask.shape
    # extend embedding with placeholder
    embeddings_ = torch.cat([-1e6*torch.ones_like(embeddings[:, :1, :]), embeddings], dim=1)
    # transform mask to index
    index = torch.arange(1, D+1).view(1, 1, -1).repeat(B, R, 1) * mask# [B, R, D]
    # batch indices
    batch_indices = torch.arange(B).view(B, 1, 1).repeat(1, R, D)
    # retrieve embeddings by index
    indexed = embeddings_[batch_indices.flatten(), index.flatten(), :].view(B, R, D, E)# [B, R, D, E]
    # return
    return indexed.max(dim=-2)

def calculate_principal_components(embeddings, num_components=3):
    """Calculates the principal components given the embedding features.

    Args:
        embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
        num_components: An integer indicates the number of principal
        components to return.

    Returns:
        A 2-D float tensor of shape `[num_pixels, num_components]`.
    """
    embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
    _, _, v = torch.svd(embeddings)
    return v[:, :num_components]


def pca(embeddings, num_components=3, principal_components=None):
    """Conducts principal component analysis on the embedding features.

    This function is used to reduce the dimensionality of the embedding.

    Args:
        embeddings: An N-D float tensor with shape with the 
        last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
        embedding features to PCA'ed space, also known as the U matrix
        from SVD. If not given, this function will calculate the
        principal_components given inputs.

    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])

    if principal_components is None:
        principal_components = calculate_principal_components(
            embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings

if __name__ == '__main__':
    # from image_utils import library_ops
    # test_params()
    hqsam = HQSam(model_type='vit_b')
    hqsam = hqsam.to('cuda')
    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').pil
    image = Im(image.crop(((image.size[0]-image.size[1]) // 2, 0, image.size[0] - (image.size[0]-image.size[1]) // 2, image.size[1]))).resize(224, 224).np
    masks = hqsam.forward(image)

    bs = len(masks)
    original = torch.from_numpy(np.array([masks[i]['segmentation'] for i in range(bs)]))
    # downscaled = torch.randn((16, 16, 768))
    output, result = find_true_indices_batched(original, 16, 16)

    from ipdb import set_trace; set_trace()

    Im(rearrange(original[:, None].repeat(1, 3, 1, 1) * 1.0, 'b c h w -> b h w c')).save('high_res_mask')
    Im(rearrange(result[:, None].repeat(1, 3, 1, 1) * 1.0, 'b c h w -> b h w c')).scale(64).save('vit_feature_mask')

    output = mask_max_pool(rearrange(downscaled, 'h w e -> () (h w) e'), rearrange(result, 'b h w -> () b (h w)'))
    output_feats = output.values

    principal_components = calculate_principal_components(embeddings, num_components)
    pca(output_feats.squeeze(0))

    show_anns(image, masks)