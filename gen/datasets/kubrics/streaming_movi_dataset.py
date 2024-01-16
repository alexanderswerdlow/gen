import autoroot

import functools
from typing import Dict, List, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tfds_preprocessing as preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from clu import deterministic_data, preprocess_spec
from einops import rearrange
from image_utils import Im
from numpy.linalg import inv
from torch.utils.data import Dataset

Array = torch.Tensor
PRNGKey = Array

def preprocess_example(features: Dict[str, tf.Tensor],
					   preprocess_strs: List[str]) -> Dict[str, tf.Tensor]:
	"""Process a single data example.

	Args:
		features: A dictionary containing the tensors of a single data example.
		preprocess_strs: List of strings, describing one preprocessing operation
			each, in clu.preprocess_spec format.

	Returns:
		Dictionary containing the preprocessed tensors of a single data example.
	"""
	all_ops = preprocessing.all_ops()
	preprocess_fn = preprocess_spec.parse("|".join(preprocess_strs), all_ops)
	return preprocess_fn(features)

def create_datasets(tfds_name, data_dir, batch_size,
	preproc_train, preproc_eval, shuffle_buffer_size,
	data_rng: PRNGKey) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	"""Create datasets for training and evaluation

	For the same data_rng and config this will return the same datasets.
	The datasets only contain stateless operations.

	Args:
		args: Configuration to use.
		data_rng: JAX PRNGKey for dataset pipeline.

	Returns:
		A tuple with the training dataset and the evaluation dataset.
	"""
	dataset_builder = tfds.builder(
		tfds_name, data_dir=data_dir)
	
	batch_dims = (batch_size,)

	train_preprocess_fn = functools.partial(
		preprocess_example, preprocess_strs=preproc_train)
	eval_preprocess_fn = functools.partial(
		preprocess_example, preprocess_strs=preproc_eval)

	train_split_name = "train"
	eval_split_name = "validation"

	# TODO: may need to do something to only run on one host
	train_split = deterministic_data.get_read_instruction_for_host(
		train_split_name, dataset_info=dataset_builder.info)
	train_ds = deterministic_data.create_dataset(
		dataset_builder,
		split=train_split,
		rng=data_rng,
		preprocess_fn=train_preprocess_fn,
		cache=False,
		shuffle_buffer_size=shuffle_buffer_size,
		batch_dims=batch_dims,
		num_epochs=None,
		shuffle=True)

	eval_split = deterministic_data.get_read_instruction_for_host(
		eval_split_name, dataset_info=dataset_builder.info, drop_remainder=False)
	eval_ds = deterministic_data.create_dataset(
		dataset_builder,
		split=eval_split,
		rng=None,
		preprocess_fn=eval_preprocess_fn,
		cache=False,
		batch_dims=batch_dims,
		num_epochs=1,
		shuffle=False,
		pad_up_to_batches="auto")

	return train_ds, eval_ds

class MOViData(Dataset):
    def __init__(self, tfds_dataset):
        self.dataset = tfds_dataset
        self.itr = iter(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        batch = jax.tree_map(np.asarray, next(self.itr))

        video = torch.from_numpy(batch['video']) # (B T H W 3)
        boxes = torch.from_numpy(batch['boxes']) # (B T maxN 4)
        # padding_mask = torch.from_numpy(batch['padding_mask'])
        # mask = torch.from_numpy(batch['mask']) if 'mask' in batch.keys() else torch.empty(0, dtype=torch.bool)
        segmentations = torch.from_numpy(batch['segmentations'])

        intrinsics = np.zeros((3, 3), dtype=np.float32)
        intrinsics[0, 0] = 293
        intrinsics[1, 1] = 293
        intrinsics[0, 2] = 128
        intrinsics[1, 2] = 128

        extrinsics = np.zeros((4, 4), dtype=np.float32)
        extrinsics[0, 0] = 1
        extrinsics[1, 2] = 1
        extrinsics[2, 2] = 1
        extrinsics[3, 3] = 1

        ret = {
            "image": video,
            'bbox': boxes,
            'intrinsics' : torch.from_numpy(intrinsics),
            'extrinsics' : torch.from_numpy(extrinsics),
            'segmentation': rearrange(segmentations, '... -> ... ()'),
        }


        return ret


if __name__ == "__main__":
    rng = jax.random.PRNGKey(2)
    # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),
    # i.e. it is safe to reduce config.max_instances to 10 for these datasets,
    # resulting in more efficient training/evaluation. We set this default to 23,
    # since MOVi-D and MOVi-E contain up to 23 objects per video. Setting
    # config.max_instances to a smaller number than the maximum number of objects
    # in a dataset will discard objects, ultimately giving different results.
    max_instances = 20 # max 
    batch_size = 2
    train_ds, eval_ds = create_datasets(
        tfds_name="movi_a/256x256:1.0.0",
        data_dir="gs://kubric-public/tfds",
        batch_size=batch_size,
        preproc_train = [
            "video_from_tfds",
            f"sparse_to_dense_annotation(max_instances={max_instances})",
            "temporal_random_strided_window(length=6)",
            # "resize_small(64)",
            "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
        ],
        preproc_eval = [
            "video_from_tfds",
            f"sparse_to_dense_annotation(max_instances={max_instances})",
            "temporal_crop_or_pad(length=24)",
            # "resize_small(64)",
            "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
        ],
        shuffle_buffer_size=8 * batch_size,
        data_rng=rng
    )

    traindata = MOViData(train_ds)
    for i in range(len(traindata)):
        data = traindata[i]
        from ipdb import set_trace; set_trace()
        Im(data[0]).save_video(f'test_{i}', fps=2)
    print('here')
