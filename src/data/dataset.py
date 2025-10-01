"""
Creates a Pytorch dataset to load the dataset used.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import B, C, S, TARGET_SIZE_IMG


class RaccoonDataset(Dataset):
    """
    Adapted Dataset class for our dataset.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
    ) -> None:
        """
        Constructor of the class.

        Args:
            img_dir: Path to the directory of the images.
            label_dir: Path to the directory of the labels.
        """

        self.imgs_dir = img_dir
        self.labels_dir = label_dir

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """

        return len(os.listdir(self.labels_dir))

    def _generate_label(self, boxes: list[list[int | float]]) -> torch.Tensor:
        """
        Generates the labels with coordinates respect to the cells, not the hole image.
        We will assume at most we can have one object per cell. Therefore, the
        dimensions of the generated label will be [S, S, 5 * B + C]. If we wanted to
        detect at most n objects the dimensions would be [S, S, n, 5 * B + C].

        Args:
            boxes: Labels with respect to the hole image. Each label has 5 elements:
                class, center_x, center_y, width and height.

        Returns:
            Labels with respect to the cells. Dimensions: [S, S, 5 * B + C].
        """

        # TODO
        label = torch.zeros((S, S, 5 * B + C))

        for box in boxes:
            class_id, x, y, width, height = box
            i, j = int(S * y), int(S * x)
            i = min(i, S - 1)  # Ensure i is within bounds
            j = min(j, S - 1)  # Ensure j is within bounds

            x_cell = S * x - j 
            y_cell = S * y - i
            width_cell = width * S
            height_cell = height * S

            # [objectness, class, x, y, w, h] for B bounding boxes
            label[i, j, class_id] = 1.0  # class
            label[i, j, C] = 1.0  # has object
            label[i, j, C + 1 : C + 5] = torch.tensor([x_cell, y_cell, width_cell, height_cell])

        return label


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image and its labels.

        Args:
            index: Index of the image and labels.

        Returns:
            The image and its labels. Dimensions of the image: [channels, height,
            width]. Dimensions of the labels: [S, S, 5 * B + C].
        """
        # Get all files and sort them
        img_files = sorted(os.listdir(self.imgs_dir))
        label_files = sorted(os.listdir(self.labels_dir))
        
        # Get the corresponding file names
        img_name = img_files[index]
        label_name = label_files[index]
        
        # Build full paths
        img_path = os.path.join(self.imgs_dir, img_name)
        label_path = os.path.join(self.labels_dir, label_name)
        
        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((TARGET_SIZE_IMG[0], TARGET_SIZE_IMG[1])),
            transforms.ToTensor()
        ])
        img = transform(img)
        
        # Parse label file
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([class_id, center_x, center_y, width, height])
    
        # Generate labels
        labels = self._generate_label(boxes)
        
        return img, labels
        