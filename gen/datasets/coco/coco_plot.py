        
from gen.datasets.coco.coco_panoptic import CocoPanoptic
from gen.datasets.coco.coco_utils import _COCO_PANOPTIC_EVAL_ID_TO_TRAIN_ID, _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID, COCO_CATEGORIES
from gen.models.cross_attn.base_inference import label_to_color_image
from gen.utils.data_defs import integer_to_one_hot


seg = integer_to_one_hot(batch.tgt_segmentation[b], num_channels=batch.valid.shape[1] + 1)[..., 1:].long().argmax(dim=-1)
instance_idx = torch.arange(batch.categories.shape[1])[batch.valid[b]]

mapping = torch.arange(seg.max() + 1)
mapping[instance_idx] = batch.categories[b][instance_idx]
seg = mapping[seg]

segmentation_map = label_to_color_image(seg, colormap=CocoPanoptic.create_label_colormap())
Im(segmentation_map).save()

import matplotlib.pyplot as plt
import numpy as np

plt.clf()
fig, ax = plt.subplots()
ax.imshow(seg, cmap='nipy_spectral')
unique_ids = np.unique(seg)

labels = [x['name'] for x in COCO_CATEGORIES]

for uid in unique_ids:
        # Find the first occurrence of this ID
        y, x = np.where(seg == uid)
        # Place the label at the first occurrence of the ID
        ax.text(x[0], y[0], labels[uid], color='white', fontsize=12,
        bbox=dict(facecolor='black', alpha=0.5))

plt.axis('off')  # Hide the axis
import io
buf = io.BytesIO()
plt.savefig(buf, format="jpeg", bbox_inches="tight")
buf.seek(0)
seg_im = Image.open(buf)
plt.close("all")

Im.concat_horizontal(seg_im, Im(((batch.tgt_pixel_values[b] + 1) / 2))).save()