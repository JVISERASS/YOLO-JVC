"""
Script to define the tests for the utilities functions.
"""

import torch

from src.constants import B, C, S
from src.utils import iou, nms, to_image_coords


def test_iou() -> None:
    """
    Test for the IoU function.
    """

    torch.manual_seed(42)
    batch = 4
    predictions = torch.randn(batch, S, S, 5 * B + C).clamp(0)
    targets = torch.randn(batch, S, S, 5 * B + C).clamp(0)
    atol = 1e-3

    iou_per_cell = iou(
        predictions[:, :, :, C + 1 : C + 5], targets[:, :, :, C + 1 : C + 5]
    )

    assert iou_per_cell.shape == torch.Size(
        [batch, S, S, 1]
    ), "Incorrect shape of the output."

    expected_max = torch.tensor(0.3730)
    expected_min = torch.tensor(0.0)
    expected_mean = torch.tensor(0.0079)
    expected_std = torch.tensor(0.0470)
    assert (
        torch.isclose(iou_per_cell.max(), expected_max, atol=atol)
        and torch.isclose(iou_per_cell.min(), expected_min, atol=atol)
        and torch.isclose(iou_per_cell.mean(), expected_mean, atol=atol)
        and torch.isclose(iou_per_cell.std(), expected_std, atol=atol)
    ), "Incorrect calculation of IoU."


def test_nms() -> None:
    """
    Test for the NMS function.
    """

    # Filters by confidence
    boxes = [
        torch.tensor([0, 0.4, 0, 0, 1, 1]),  # < threshold_confidence
        torch.tensor([0, 0.6, 0, 0, 1, 1]),  # >= threshold_confidence
    ]
    result = nms(boxes, threshold_confidence=0.5)
    assert len(result) == 1, "Incorrect filter by confidence."
    assert torch.equal(result[0], boxes[1]), "Incorrect filter by confidence."

    # Overlapping boxes
    boxes = [
        torch.tensor([0, 0.9, 0, 0, 1, 1]),  # chosen_box
        torch.tensor([0, 0.8, 0, 0, 1, 1]),  # overlaps -> should be removed
        torch.tensor([1, 0.7, 0, 0, 1, 1]),  # different class -> keep
    ]
    result = nms(boxes, threshold_confidence=0.5, threshold_repeated=0.5)
    assert len(result) == 2, "Incorrect removing boxes."
    kept_classes = set(box[0].item() for box in result)
    assert 0 in kept_classes and 1 in kept_classes, "Incorrect removing boxes."

    # One single box
    box = torch.tensor([0, 0.9, 0, 0, 1, 1])
    result = nms([box])
    assert len(result) == 1, "Incorrect for one box."
    assert torch.equal(result[0], box), "Incorrect for one box."


def test_to_image_coords() -> None:
    """
    Test for the function that transforms coordinates.
    """

    img_w, img_h = 224, 224
    boxes = torch.zeros((S, S, 5 * B + C))
    i, j = 3, 4  # selected cell
    # Fill the cell with some values
    boxes[i, j, 0] = 0.8
    boxes[i, j, 1:6] = torch.tensor([0.9, 0.5, 0.5, 0.2, 0.3])
    boxes[i, j, 6:11] = torch.tensor([0.6, 0.1, 0.2, 0.1, 0.1])

    result = to_image_coords(boxes, img_w, img_h)

    assert len(result) == S * S * B, "Incorrect shape."
    non_zero_boxes = [b for b in result if b[1] > 0]
    assert len(non_zero_boxes) == 2, "Incorrect shape."

    for box in non_zero_boxes:
        box = non_zero_boxes[0]
        # Class prob and p(obj)
        assert torch.isclose(box[0], torch.tensor(0.8)), "Incorrect class."
        assert torch.isclose(box[1], torch.tensor(0.9)), "Incorrect p(obj)."
        cell_size = 1.0 / S
        expected_cx = (j + 0.5) * cell_size * img_w
        expected_cy = (i + 0.5) * cell_size * img_h
        expected_w = 0.2 * cell_size * img_w
        expected_h = 0.3 * cell_size * img_h
        # Coordinates
        assert torch.isclose(
            box[2], torch.tensor(expected_cx)
        ), "Incorrect coordinates."
        assert torch.isclose(
            box[3], torch.tensor(expected_cy)
        ), "Incorrect coordinates."
        assert torch.isclose(box[4], torch.tensor(expected_w)), "Incorrect coordinates."
        assert torch.isclose(box[5], torch.tensor(expected_h)), "Incorrect coordinates."
