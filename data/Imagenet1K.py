from torch.utils.data import Dataset
from PIL import Image
import os


class Imagenet1KTest(
    Dataset,
):
    def __init__(self, dataset_dir, split, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = os.path.join(dataset_dir, "ILSVRC2012", split)
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
