from __future__ import annotations
from pathlib import Path

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[9]
                images.append(image_name)
    return images

if __name__ == "__main__":
    semantic_root = Path('/home/aswerdlo/data/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/data')

    all_image_file_paths = []
    for scene_dir in semantic_root.iterdir():
        if scene_dir.is_dir():
            image_names = read_images_text(scene_dir / 'iphone' / 'colmap' / 'images.txt')
            image_file_paths = [scene_dir / 'iphone' / 'rgb' / (image_name) for image_name in image_names]
            all_image_file_paths.extend(image_file_paths)

    # write to file
    all_image_file_paths = [x for x in all_image_file_paths if x.exists()]
    with open(Path(__file__).parent / 'all_image_file_paths.txt', 'w') as f:
        for image_file_path in all_image_file_paths:
            f.write(str(image_file_path) + '\n')