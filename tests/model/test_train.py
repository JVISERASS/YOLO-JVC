"""
Script to define the tests for the training.
"""

import os

import torch
from torch.utils.data import DataLoader

from model.train import Trainer
from src.constants import BATCH_SIZE

TRAINER = Trainer()


def test_get_loaders() -> None:
    """
    Test for the _get_loaders function.
    """

    assert isinstance(TRAINER.train_loader, DataLoader) and isinstance(
        TRAINER.test_loader, DataLoader
    ), "Incorrect type for the loaders."
    assert (
        len(TRAINER.train_loader) * BATCH_SIZE == 160
        and len(TRAINER.test_loader) * BATCH_SIZE == 32
    ), "Incorrect length of the loaders."


def test_make_epoch_train() -> None:
    """
    Test for the _make_epoch_train function. It is not exhaustive. It just checks that
    the code works, but it should optimize the parameters for each batch of the train
    loader.
    """

    # This part is just to initialize the parameters of the model, as if we use lazy
    # modules, like LazyLinear, they are not generated until a forward pass is done.
    for imgs, _ in TRAINER.train_loader:
        TRAINER.model(imgs)
        break

    prev_params = [p.clone().detach() for p in TRAINER.model.parameters()]
    TRAINER._make_epoch_train()
    new_params = [p.clone().detach() for p in TRAINER.model.parameters()]

    for loss_type in ["coord", "obj", "noobj", "class", "loss"]:
        assert len(TRAINER.losses[loss_type]) > 0, "You have to append the losses."

    for i, (old, new) in enumerate(zip(prev_params, new_params)):
        assert (
            torch.norm(new - old).item() > 0
        ), f"Parameters were not updated in layer {i}"


def test_fit() -> None:
    """
    Test for the training of the model.
    """

    path_weights = "weights/YOLO_test.pt"
    path_images = "images/training/test_evolution"
    TRAINER.fit(n_epochs=2, path_weights=path_weights, path_images=path_images)

    assert os.path.exists(path_weights), "Weights of the model should be saved."
    assert os.path.exists(f"{path_images}_loss_0.png") and os.path.exists(
        f"{path_images}_map.png"
    ), "Evolution of the loss and the mAP should be saved as Figures."

    os.remove(f"{path_images}_loss_0.png")
    os.remove(f"{path_images}_map.png")
    os.remove(path_weights)


def test_test() -> None:
    """
    Test for the test of the model.
    """

    assert isinstance(TRAINER.test(), float), "The test should return a float."
