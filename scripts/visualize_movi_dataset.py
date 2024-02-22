"""Visualize dataset entries for debugging.
"""
import os

from transformers import AutoTokenizer, CLIPTokenizer
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
import cv2
from sklearn.manifold import TSNE
import PIL.Image as Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from gen.models.utils import get_model_from_cfg
from gen.configs.base import BaseConfig
from train import Trainer


class DebugTrainer(Trainer):

    def __init__(self, cfg: BaseConfig, accelerator: Accelerator):
        # only instantiate the dataloader
        self.cfg = cfg
        self.train_dataloader: DataLoader = None
        self.validation_dataloader: DataLoader = None
        self.tokenizer: CLIPTokenizer = None
        self.accelerator: Accelerator = None
        self.init_models()
        self.init_dataloader()

    def init_models(self):
        # TODO: Define a better interface for different models once we get a better idea of the requires inputs/outputs
        # Right now we just conditionally call methods in the respective files based on the model_type enum
        assert is_xformers_available()

        self.models = []
        model = get_model_from_cfg(self.cfg)
        self.models.append(model)
        self.model = model
        self.tokenizer = model.tokenizer


def get_rotation_and_cropped_objects(batch):
    # disregard the first channel, which is the background
    segmentation = batch['disc_segmentation'][:, :, 1:].numpy()
    image = batch['disc_pixel_values'].numpy().transpose(1, 2, 0)

    std = np.array((0.26862954, 0.26130258, 0.27577711))
    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    image = (image * std) + mean
    image = (image * 255).astype(np.uint8)

    mask = segmentation.sum(axis=(0, 1)) > 16
    if mask.any():
        # get quat and check
        quats = batch['quaternions'][mask]
        assert (np.abs(np.linalg.norm(quats) - 1).max() < 1e-5).all()
        rot_mat = R.from_quat(quats).as_matrix().reshape(1, 9)

        # get cropped objects
        objects = []
        for i in range(mask.shape[0]):
            if mask[i]:
                ys, xs = np.where(segmentation[:, :, i])
                min_y, max_y = ys.min(), ys.max()
                min_x, max_x = xs.min(), xs.max()
                curr_obj = image[min_y:max_y, min_x:max_x, :]
                curr_obj = cv2.resize(curr_obj, (64, 64))
                objects.append(curr_obj)
        objects = np.stack(objects, axis=0)
    else:
        rot_mat = np.zeros((0, 9))
        objects = np.zeros((0, 64, 64, 3))

    return rot_mat, objects


def plot_rotation_on_2d_image(train_feats, train_images, val_feats, val_images,
                              is_tsne=True, save_name='tsne.png'):
    # prepare canvas to put all images
    num_image_per_side = 200
    h, w = train_images[0][0].shape[:2]
    canvas = np.zeros(
        (num_image_per_side * h, num_image_per_side * w, 3), dtype=np.uint8
    )

    all_feats = np.concatenate(train_feats + val_feats, axis=0)
    all_images = np.concatenate(train_images + val_images, axis=0)
    if is_tsne:
        # get image location with 2D t-SNE
        tsne = TSNE(n_components=2, random_state=0)

        all_feats_2d = tsne.fit_transform(all_feats)
    else:
        assert all_feats.shape[-1] == 3
        all_feats_2d = all_feats[..., :2].copy()

        for i in range(len(all_images)):
            cv2.putText(all_images[i],
                        '{:.1f}'.format(all_feats[i, 2]),
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
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
    val_feats_tsne = all_feats_2d[len(train_feats):]

    train_images = all_images[:len(train_images)]
    val_images = all_images[len(train_images):]

    for feat, img in zip(train_feats_tsne, train_images):
        st_y, st_x = (feat - np.array([h, w]) * 0.5).astype(np.int32)
        img[:2, :] = (255, 0, 0)
        img[-2:, :] = (255, 0, 0)
        img[:, :2] = (255, 0, 0)
        img[:, -2:] = (255, 0, 0)
        canvas[st_y:st_y + h, st_x:st_x + w] = img

    for feat, img in zip(val_feats_tsne, val_images):
        st_y, st_x = (feat - np.array([h, w]) * 0.5).astype(np.int32)
        img[:2, :] = (0, 255, 0)
        img[-2:, :] = (0, 255, 0)
        img[:, :2] = (0, 255, 0)
        img[:, -2:] = (0, 255, 0)
        canvas[st_y:st_y + h, st_x:st_x + w] = img

    Image.fromarray(canvas, mode='RGB').save(save_name)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: BaseConfig):
    with open_dict(cfg):
        cfg.cwd = str(get_original_cwd())

    weight_dtype = torch.bfloat16
    cfg.trainer.dtype = str(weight_dtype)

    # Hydra automatically changes the working directory, but we stay at the project directory.
    os.chdir(cfg.cwd)

    trainer = DebugTrainer(cfg=cfg, accelerator=None)

    viz = False
    viz = True
    num_split = 4
    start_ind = int(os.environ.get('LOCAL_RANK', 0))
    print('----------', start_ind, '--------------')

    try:
        if viz:
            train_rots, train_objects, val_rots, val_objects = [], [], [], []
            for i in range(num_split):
                cached = np.load(f'cached_{i}.npy', allow_pickle=True).item()
                cached['train_rots'] = [R.from_matrix(m.reshape(1, 3, 3)).as_euler("ZXY", degrees=True) for m in cached['train_rots'] if m.shape[0] > 0]
                train_rots.extend(cached['train_rots'])
                train_objects.extend([m for m in cached['train_objects'] if m.shape[0] > 0])
                cached['val_rots'] = [R.from_matrix(m.reshape(1, 3, 3)).as_euler("ZXY", degrees=True) for m in cached['val_rots'] if m.shape[0] > 0]
                val_rots.extend(cached['val_rots'])
                val_objects.extend([m for m in cached['val_objects'] if m.shape[0] > 0])

            splits = np.linspace(-180, 180, 19)
            for i in range(len(splits) - 1):
                curr_train_rots, curr_train_objects = [], []
                for j in range(len(train_rots)):
                    if splits[i] <= train_rots[j][0, -1] < splits[i + 1]:
                        curr_train_rots.append(train_rots[j])
                        curr_train_objects.append(train_objects[j])

                curr_val_rots, curr_val_objects = [], []
                for j in range(len(val_rots)):
                    if splits[i] <= val_rots[j][0, -1] < splits[i + 1]:
                        curr_val_rots.append(val_rots[j])
                        curr_val_objects.append(val_objects[j])
                plot_rotation_on_2d_image(curr_train_rots, curr_train_objects, curr_val_rots, curr_val_objects, is_tsne=False, save_name=f'euler_angle_{i}.png')
        else:
            val_rots = []
            val_objects = []
            for i in tqdm(range(start_ind, len(trainer.validation_dataloader.dataset), num_split)):
                batch = trainer.validation_dataloader.dataset[i]
                rots, objects = get_rotation_and_cropped_objects(batch)
                val_rots.append(rots)
                val_objects.append(objects)

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
            # np.save(f'cached_{start_ind}.npy', cached)
            np.save(f'cached_{start_ind}.npy', cached)

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