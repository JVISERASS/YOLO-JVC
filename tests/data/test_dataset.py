"""
Script to define the tests for the adapted Dataset class.
"""

import torch

from src.constants import TARGET_SIZE_IMG, B, C, S
from src.data.dataset import RaccoonDataset

IMGS_DIR = "data/images_resized/train"
LABELS_DIR = "data/labels_resized/train"
DATASET = RaccoonDataset(IMGS_DIR, LABELS_DIR)


def test_generate_label() -> None:
    """
    Test for the _generate_label function. The first two bounding boxes refer to the
    same cell, so only one object should be detected in that cell, but two in the hole
    image.
    """

    boxes = [[1, 0.5, 0.5, 0.2, 0.3], [1, 0.5, 0.5, 0.2, 0.3], [1, 0, 0, 0.2, 0.3]]
    label = DATASET._generate_label(boxes)

    # First bounding box
    i, j = int(S * boxes[0][1]), int(S * boxes[0][2])
    assert label.shape == (S, S, 5 * B + C), "Incorrect shape for the label."
    assert label[i, j, C] == 1.0, "Incorrect 'has object' for the label."
    assert label[i, j, 1] == 1.0, "Incorrect class for the label"
    # Second bounding box
    ii, jj = int(S * boxes[2][1]), int(S * boxes[2][2])
    assert label.shape == (S, S, 5 * B + C), "Incorrect shape for the label."
    assert label[ii, jj, C] == 1.0, "Incorrect 'has object' for the label."
    assert label[ii, jj, 1] == 1.0, "Incorrect class for the label"
    # In this cell nothing should be detected
    assert (
        label[-1, -1, C] == 0.0 and label[-1, -1, 1] == 0.0
    ), "Values in this cell should be zero."

    # Relative coordinates of the cell (we just check the first bounding box)
    x_cell = S * boxes[0][1] - j
    y_cell = S * boxes[0][2] - i
    width_cell = boxes[0][3] * S
    height_cell = boxes[0][4] * S
    coords = label[i, j, C + 1 : C + 5]
    expected_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])
    assert torch.allclose(
        coords, expected_coords
    ), "Incorrect coordinates for the label."


def test_getitem() -> None:
    """
    Test for the __getitem__ function.
    """

    index = 0
    img, labels = DATASET[index]

    assert isinstance(img, torch.Tensor) and img.shape == torch.Size(
        [3, TARGET_SIZE_IMG[0], TARGET_SIZE_IMG[1]]
    ), "Incorrect format for the image."
    assert isinstance(labels, torch.Tensor) and labels.shape == torch.Size(
        [S, S, 5 * B + C]
    ), "Incorrect format for the labels."
