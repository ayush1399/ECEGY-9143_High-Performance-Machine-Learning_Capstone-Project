import os
from time import sleep
import h5py

import numpy as np

from PIL import Image

subsets = {
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
    "extra": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
    "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
    "weather": ["brightness", "fog", "frost", "snow"],
}

dirs = list()
for key in subsets.keys():
    for val in subsets[key]:
        dirs += [os.path.join(key, val)]

create_dataset_args = {"compression": "gzip", "compression_opts": 9}

for dir in dirs:
    for i in range(1, 5 + 1):
        hdf5_path = os.path.join(dir, f"{i}.hdf5")

        if os.path.exists(hdf5_path):
            print(f"Removing: {hdf5_path}")
            os.remove(hdf5_path)

        with h5py.File(hdf5_path, "w") as hdf5_file:
            subdirs = sorted(os.listdir(os.path.join(dir, str(i))))
            for subdir in subdirs:
                dataset = hdf5_file.create_dataset(
                    subdir, (50, *(224, 224, 3)), dtype=np.uint8
                )
                print(f"In {hdf5_path} Created dataset: {subdir}")
                subdir = os.path.join(dir, str(i), subdir)

                files = sorted(os.listdir(subdir))
                files = [f for f in files if f.endswith(".JPEG")]
                assert len(files) == 50

                for idx, f in enumerate(files):
                    with Image.open(os.path.join(subdir, f)) as img:
                        img_array = np.array(img, dtype=np.uint8)  # / 255.0
                        dataset[idx] = img_array

            print(f'Saved: {os.path.join(dir, f"{i}.hdf5")}\n')
            sleep(1)
