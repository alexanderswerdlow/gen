"""Visualize dataset entries for debugging.
"""
import os

import cv2
import hydra
import numpy as np
import PIL.Image as Image
import torch
from accelerate import Accelerator
from hydra.utils import get_original_cwd
from omegaconf import open_dict
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTokenizer

from gen.configs.base import BaseConfig
from gen.datasets.base_dataset import AbstractDataset, Split
from train import Trainer


class DebugTrainer(Trainer):

    def __init__(self, cfg: BaseConfig, accelerator: Accelerator):
        self.cfg = cfg
        self.train_dataloader: DataLoader = None
        self.validation_dataloader: DataLoader = None
        self.tokenizer: CLIPTokenizer = None
        self.accelerator: Accelerator = None
        self.init_models()
        self.init_dataloader()

    def init_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    def init_dataloader(self):
        self.train_dataloader_holder: AbstractDataset = hydra.utils.instantiate(self.cfg.dataset.train_dataset, _recursive_=True)(
            cfg=self.cfg, split=Split.TRAIN, tokenizer=self.tokenizer, accelerator=self.accelerator
        )
        self.train_dataloader: DataLoader = self.train_dataloader_holder.get_dataloader()
        assert len(self.train_dataloader) > 0

def get_ortho6d_from_rotation_matrix(matrix):
    # Noe the orhto6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
    return ortho6d

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

def resize_with_padding(input_array, target_size=(64, 64)):
    image = Image.fromarray(input_array)
    input_aspect = image.width / image.height
    target_aspect = target_size[0] / target_size[1]
    if input_aspect > target_aspect:
        # Width is the limiting dimension
        resize_width = target_size[0]
        resize_height = round(resize_width / input_aspect)
    else:
        # Height is the limiting dimension
        resize_height = target_size[1]
        resize_width = round(resize_height * input_aspect)
    
    image_resized = image.resize((resize_width, resize_height), Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    top_left_x = (target_size[0] - resize_width) // 2
    top_left_y = (target_size[1] - resize_height) // 2
    new_image.paste(image_resized, (top_left_x, top_left_y))
    result_array = np.array(new_image)
    return result_array

def get_rotation_and_cropped_objects(batch):
    # disregard the first channel, which is the background
    segmentation = batch['disc_segmentation'][:, :, 1:].numpy()
    image = batch['disc_pixel_values'].numpy().transpose(1, 2, 0)

    std = np.array((0.26862954, 0.26130258, 0.27577711))
    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    image = (image * std) + mean
    image = (image * 255).astype(np.uint8)
    target_size = (96, 96)

    mask = segmentation.sum(axis=(0, 1)) > 768
    if mask.any():
        quats = batch['quaternions'][mask]
        assert (np.abs(np.linalg.norm(quats) - 1).max() < 1e-5).all()
        rot_mat = R.from_quat(quats).as_matrix().reshape(1, 9)

        # get cropped objects
        objects = []
        for i in range(mask.shape[0]):
            if mask[i]:
                ys, xs = np.where(segmentation[:, :, i])
                min_y, max_y = max(0, ys.min() - 8), min(ys.max() + 8, image.shape[0])
                min_x, max_x = max(0, xs.min() - 8), min(xs.max() + 8, image.shape[1])
                curr_obj = image[min_y:max_y, min_x:max_x, :]
                curr_obj = resize_with_padding(curr_obj, target_size=target_size)
                objects.append(curr_obj)
        objects = np.stack(objects, axis=0)
    else:
        rot_mat = np.zeros((0, 9))
        objects = np.zeros((0, *target_size, 3))

    return rot_mat, objects

def inner_product_quaternion_distance(x, y):
    q_1 = R.from_matrix(x.reshape(3, 3)).as_quat()
    q_2 = R.from_matrix(y.reshape(3, 3)).as_quat()
    return 1 - np.abs(np.dot(q_1, q_2))


def plot_on_2d_image(train_feats, train_images, val_feats, val_images,
                              is_tsne=True, save_name='tsne.png'):
    # prepare canvas to put all images
    num_image_per_side = 200
    h, w = train_images[0][0].shape[:2]
    canvas = np.zeros(
        (num_image_per_side * h, num_image_per_side * w, 3), dtype=np.uint8
    )

    all_feats = np.concatenate(train_feats + val_feats, axis=0)
    all_images = np.concatenate(train_images + val_images, axis=0)
    
    viz_pca = True
    if viz_pca:
        if is_tsne:
            rot_6d = get_ortho6d_from_rotation_matrix(all_feats.reshape(-1, 3, 3))
        else:
            rot_6d = get_ortho6d_from_rotation_matrix(R.from_euler(seq='zyx', angles=all_feats, degrees=False).as_matrix())
        rot_pca = pca(rot_6d)
        rot_pca = (rot_pca - rot_pca.min()) / (rot_pca.max() - rot_pca.min())

    if is_tsne:
        # get image location with 2D t-SNE
        tsne = TSNE(n_components=2, random_state=0)

        all_feats_2d = tsne.fit_transform(all_feats)
    else:
        assert all_feats.shape[-1] == 3
        all_feats_2d = all_feats[..., :2].copy()

        for i in range(len(all_images)):
            cv2.putText(all_images[i],
                        f'{all_feats[i, 0]:.1f},{all_feats[i, 1]:.1f},{all_feats[i, 2]:.1f}',
                        (5, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 255),
                        1,
                        2)

    min_y, min_x = all_feats_2d.min(axis=0)
    max_y, max_x = all_feats_2d.max(axis=0)

    all_feats_2d -= np.array([min_y, min_x])
    all_feats_2d /= np.array([max_y - min_y, max_x - min_x])
    all_feats_2d *= np.array(canvas.shape[:2]) - np.array([h, w])
    all_feats_2d += np.array([h, w]) * 0.5

    train_feats_tsne = all_feats_2d[:len(train_feats)]
    train_images = all_images[:len(train_images)]

    for idx, (feat, img) in enumerate(zip(train_feats_tsne, train_images)):
        st_y, st_x = (feat - np.array([h, w]) * 0.5).astype(np.int32)
        color_ = (rot_pca[idx] * 255).to(dtype=torch.uint8).tolist()
        img[:4, :] = color_
        img[-4:, :] = color_
        img[:, :4] = color_
        img[:, -4:] = color_
        canvas[st_y:st_y + h, st_x:st_x + w] = img

    Image.fromarray(canvas, mode='RGB').save(save_name)

@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: BaseConfig):
    with open_dict(cfg):
        cfg.cwd = str(get_original_cwd())

    weight_dtype = torch.bfloat16
    cfg.trainer.dtype = str(weight_dtype)
    os.chdir(cfg.cwd)

    trainer = DebugTrainer(cfg=cfg, accelerator=None)

    load = False
    num_split = 1
    start_ind = os.environ.get('LOCAL_RANK', 0)
    print('------------', start_ind, '---------------------')

    try:
        proj_name = os.environ.get('proj', 'cached_')
        if load:
            train_rots, train_objects, val_rots, val_objects = [], [], [], []
            for i in range(num_split):
                cached = np.load(f'output/{proj_name}{i}.npy', allow_pickle=True).item()
                train_rots.extend(cached['train_rots'])
                train_objects.extend(cached['train_objects'])
                val_rots.extend(cached['val_rots'])
                val_objects.extend(cached['val_objects'])
        else:
            val_rots = []
            val_objects = []
            train_rots = []
            train_objects = []
            for i in tqdm(range(start_ind, len(trainer.train_dataloader.dataset), num_split)):
                batch = trainer.train_dataloader.dataset[i]
                rots, objects = get_rotation_and_cropped_objects(batch)
                train_rots.append(rots)
                train_objects.append(objects)

            cached = {
                'train_rots': train_rots,
                'train_objects': train_objects,
                'val_rots': val_rots,
                'val_objects': val_objects
            }
            np.save(f'output/{proj_name}{start_ind}.npy', cached)

        save_name = os.environ.get('proj', 'tsne')
        plot_on_2d_image(train_rots, train_objects, val_rots, val_objects, save_name=f'output/{save_name}_tsne.png', is_tsne=True)
        print('Finished saving to', f'output/{save_name}_tsne.png')

        train_rots, train_objects, val_rots, val_objects = [], [], [], []
        for i in range(num_split):
            cached = np.load(f'output/{proj_name}{i}.npy', allow_pickle=True).item()
            cached['train_rots'] = [R.from_matrix(m.reshape(1, 3, 3)).as_euler("zyx", degrees=False) for m in cached['train_rots'] if m.shape[0] > 0]
            train_rots.extend(cached['train_rots'])
            train_objects.extend([m for m in cached['train_objects'] if m.shape[0] > 0])
            cached['val_rots'] = [R.from_matrix(m.reshape(1, 3, 3)).as_euler("zyx", degrees=False) for m in cached['val_rots'] if m.shape[0] > 0]
            val_rots.extend(cached['val_rots'])
            val_objects.extend([m for m in cached['val_objects'] if m.shape[0] > 0])

        save_name = os.environ.get('proj', 'euler') 
        plot_on_2d_image(train_rots, train_objects, val_rots, val_objects, is_tsne=False, save_name=f'output/{save_name}_euler.png')
        print('Finished saving to', f'output/{save_name}_euler.png')

    except Exception as e:
        import sys
        import traceback

        import ipdb

        traceback.print_exc()
        ipdb.post_mortem(e.__traceback__)
        sys.exit(1)
        raise
    finally:
        pass

if __name__ == "__main__":
    main()