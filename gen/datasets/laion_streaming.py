from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
from datasets import load_dataset
import urllib
import PIL

USER_AGENT = f"datasets"

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

num_threads = 1

def fetch_images(batch, timeout=None, retries=0):
    print(batch.keys())
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["URL"]))
    return batch



dset = load_dataset("laion/laion2B-en", split='train', streaming=True)
dset = dset.map(fetch_images, batched=True, batch_size=2)
import torch
# dataloader = DataLoader(my_iterable_dataset, batch_size=32, num_workers=4)
# dset.n_shards = 128
dataloader = torch.utils.data.DataLoader(dset, batch_size=2)
output = next(iter(dataloader))
print()