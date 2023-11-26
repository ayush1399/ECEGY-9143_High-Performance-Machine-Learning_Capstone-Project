from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torch
import os

num_workers = os.cpu_count()
num_workers = max(1, num_workers) if num_workers is not None else 1


class I1KEvalBase(Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        self.image_dir = os.path.join(dataset_dir, "ILSVRC2012", split)
        self.transform = transform
        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if os.path.isfile(os.path.join(self.image_dir, f))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class I1KTest(
    I1KEvalBase,
):
    def __init__(self, dataset_dir, transform=None):
        super().__init__(dataset_dir, "test", transform)


class I1KVal(
    I1KEvalBase,
):
    def __init__(self, dataset_dir, transform=None):
        super().__init__(dataset_dir, "val", transform)
        with open(
            os.path.join(dataset_dir, "ILSVRC2012_validation_ground_truth.txt"), "r"
        ) as f:
            self.labels = f.readlines()
        self.labels = [int(label.strip()) for label in self.labels]

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        return image, self.labels[idx]

    @staticmethod
    def eval_model(
        model,
        root=".",
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
    ):
        model.eval()
        correct = 0
        total = 0

        dataset = I1KVal(root, transforms)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers  # type: ignore
        )

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        return correct / total
