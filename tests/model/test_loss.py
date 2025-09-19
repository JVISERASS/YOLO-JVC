"""
Script to define the tests for the YOLO Loss.
"""

import torch

from src.model.loss import YOLOLoss
from src.constants import B, C, S

torch.manual_seed(42)

BATCH = 4
PREDICTIONS = torch.randn(BATCH, S, S, 5 * B + C)
TARGETS = torch.randn(BATCH, S, S, 5 * B + C).clamp(0)
ONE_I_OBJ = torch.randint(0, 2, (BATCH, S, S, 1))
BEST_BOX = torch.randint(0, 2, (BATCH, S, S, 1))
YOLO_LOSS = YOLOLoss()
ATOL = 1e-3


def test_coord_loss() -> None:
    """
    Test for the coordinates loss.
    """

    expected = torch.tensor(3.0959)

    assert torch.isclose(
        YOLO_LOSS._coordinates_loss(PREDICTIONS, TARGETS, ONE_I_OBJ, BEST_BOX),
        expected,
        atol=ATOL,
    ), "Incorrect calculation of the coordinates loss."


def test_obj_loss() -> None:
    """
    Test for the object loss.
    """

    expected = torch.tensor(0.6191)

    assert torch.isclose(
        YOLO_LOSS._object_loss(PREDICTIONS, TARGETS, ONE_I_OBJ, BEST_BOX),
        expected,
        atol=ATOL,
    ), "Incorrect calculation of the object loss."


def test_no_obj_loss() -> None:
    """
    Test for the no object loss.
    """

    expected = torch.tensor(0.3153)

    assert torch.isclose(
        YOLO_LOSS._no_object_loss(PREDICTIONS, TARGETS, ONE_I_OBJ),
        expected,
        atol=ATOL,
    ), "Incorrect calculation of the no object loss."


def test_class_loss() -> None:
    """
    Test for the classification loss.
    """

    expected = torch.tensor(0.7886)

    assert torch.isclose(
        YOLO_LOSS._class_loss(PREDICTIONS, TARGETS, ONE_I_OBJ),
        expected,
        atol=ATOL,
    ), "Incorrect calculation of the classification loss."


def test_forward() -> None:
    """
    Test for the forward pass.
    """

    expected = (
        torch.tensor(2.5873),
        torch.tensor(1.4043),
        torch.tensor(0.4504),
        torch.tensor(0.7253),
        torch.tensor(5.1673),
    )

    coord_loss, object_loss, no_object_loss, class_loss, loss = YOLO_LOSS(
        PREDICTIONS, TARGETS
    )

    for expected_loss, actual_loss in zip(
        expected, [coord_loss, object_loss, no_object_loss, class_loss, loss]
    ):
        assert torch.isclose(
            expected_loss, actual_loss, atol=ATOL
        ), "Incorrect calculation of the forward pass."
