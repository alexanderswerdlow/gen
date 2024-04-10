import os
from PIL import Image

def crop_and_resize(image, target_width=512, target_height=512):
    image_aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    if image_aspect_ratio > target_aspect_ratio:
        new_height = target_height
        new_width = int(image.width * (new_height / image.height))
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)
        left = (new_width - target_width) // 2
        right = left + target_width
        cropped_image = resized_image.crop((left, 0, right, target_height))
    else:
        new_width = target_width
        new_height = int(image.height * (new_width / image.width))
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)
        top = (new_height - target_height) // 2
        bottom = top + target_height
        cropped_image = resized_image.crop((0, top, target_width, bottom))

    return cropped_image

def process_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                processed_image = crop_and_resize(image)
                new_filename = os.path.splitext(file)[0] + "_512.png"
                new_image_path = os.path.join(root, new_filename)
                processed_image.save(new_image_path)

# Example usage
process_images("/mnt/ssd/aswerdlo/data/projects/katefgroup/aswerdlo/datasets/hypersim/train/rgb")
process_images("/mnt/ssd/aswerdlo/data/projects/katefgroup/aswerdlo/datasets/hypersim/valid/rgb")
