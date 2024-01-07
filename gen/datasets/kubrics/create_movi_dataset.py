import autoroot

import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import typer

from gen import MOVI_DATASET_PATH

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset_dir: Path = MOVI_DATASET_PATH,
    dataset_name: str = "movi_e",
    specific_split: Optional[str] = None,
):  
    splits = (specific_split, ) if specific_split else ("train", "validation")
    for split in splits:
        out_dir = dataset_dir / dataset_name / split
        ds, info = tfds.load(dataset_name, data_dir="gs://kubric-public/tfds", with_info=True)
        train_iter = iter(tfds.as_numpy(ds[split]))

        example = next(train_iter)

        count = 0

        def crop_or_pad(t, size, value, allow_crop=True):
            pad_size = tf.maximum(size - tf.shape(t)[0], 0)
            t = tf.pad(
                t, ((0, pad_size),) + ((0, 0),) * (len(t.shape) - 1),  # pytype: disable=attribute-error  # allow-recursive-types
                constant_values=value)
            if allow_crop:
                t = t[:size]
            return t

        os.makedirs(out_dir, exist_ok=True)
        NOTRACK_BOX = (0., 0., 0., 0.)  # No-track bounding box for padding.
        while example:
            rgb = example["video"]
            segment = example["segmentations"]

            video_name = 'video_{}'.format(str(count).zfill(4))
            bbox = example['instances']['bboxes']
            max_instances = 10
            num_frames = tf.shape(rgb)[0]
            frames = example["instances"]["bbox_frames"]
            num_tracks = tf.shape(frames)[0]

            def densify_boxes(n):
                boxes_n = tf.tensor_scatter_nd_update(
                    tf.tile(tf.constant(NOTRACK_BOX)[tf.newaxis], (num_frames, 1)),
                    frames[n][:, tf.newaxis], bbox[n])
                return boxes_n

            bbox = tf.map_fn(
                densify_boxes,
                tf.range(tf.minimum(num_tracks, max_instances)),
                fn_output_signature=tf.float32)

            bbox = tf.transpose(crop_or_pad(bbox, max_instances, NOTRACK_BOX[0]), (1, 0, 2))

            os.makedirs(out_dir / video_name, exist_ok=True)

            np.save(out_dir / video_name / 'rgb.npy', rgb)
            np.save(out_dir / video_name / 'segment.npy', segment)
            np.save(out_dir / video_name / 'bbox.npy', bbox)

            if count % 200 == 0:
                print(count)

            count += 1
            try:
                example = next(train_iter)
            except StopIteration:
                print(f"Finished split {split}")
                continue


if __name__ == "__main__":
    app()
