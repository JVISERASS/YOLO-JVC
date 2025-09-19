"""
Script to define the tests for the YOLO model.
"""

import torch
from torch import nn

from src.model.model import YOLO, CNNBlock
from src.constants import B, C, S

MODEL = YOLO()


def test_create_cnn() -> None:
    """
    Test for the creation of the CNN part.
    """

    architecture: list[tuple[int, int, int, int] | str] = [
        (7, 8, 2, 3),
        "M",
        (3, 16, 1, 1),
        "M",
    ]
    in_channels = 3

    cnn = MODEL._create_cnn(architecture, in_channels)

    assert isinstance(cnn, nn.Sequential)
    assert len(cnn) == len(architecture)

    # 1. CNNBlock
    assert isinstance(cnn[0], CNNBlock)
    assert cnn[0].conv.in_channels == 3
    assert cnn[0].conv.out_channels == 8
    assert cnn[0].conv.kernel_size == (7, 7)
    assert cnn[0].conv.stride == (2, 2)
    assert cnn[0].conv.padding == (3, 3)

    # 2. MaxPool2d
    assert isinstance(cnn[1], nn.MaxPool2d)
    assert cnn[1].kernel_size == (2, 2)
    assert cnn[1].stride == (2, 2)

    # 3. CNNBlock
    assert isinstance(cnn[2], CNNBlock)
    assert cnn[2].conv.in_channels == 8
    assert cnn[2].conv.out_channels == 16
    assert cnn[2].conv.kernel_size == (3, 3)
    assert cnn[2].conv.stride == (1, 1)
    assert cnn[2].conv.padding == (1, 1)

    # 4. MaxPool2d
    assert isinstance(cnn[3], nn.MaxPool2d)
    assert cnn[3].kernel_size == (2, 2)
    assert cnn[3].stride == (2, 2)


def test_create_mlp() -> None:
    """
    Test for the creation of the MLP part.
    """

    head = MODEL._create_mlp()

    assert isinstance(head, nn.Sequential)
    assert len(head) >= 3
    assert isinstance(head[0], nn.Flatten)
    assert isinstance(head[1], (nn.LazyLinear, nn.Linear))
    assert isinstance(head[-1], nn.Linear)
    assert head[-1].out_features == S * S * (5 * B + C)


def test_forward() -> None:
    """
    Test for the forward pass.
    """

    batch = 4
    channels = 3
    height = 224
    width = 224
    x = torch.randn(batch, channels, height, width)
    y = MODEL(x)
    assert y.shape == torch.Size([batch, S, S, 5 * B + C])
