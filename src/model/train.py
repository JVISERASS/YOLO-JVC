"""
Script to train the model.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import BATCH_SIZE, EPOCHS, LR, PATH_MODEL
from src.data.dataset import RaccoonDataset
from src.model.loss import YOLOLoss
from src.model.model import YOLO
from src.utils import mean_average_precision
import os


class Trainer:
    """
    Class of the trainer of the model.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        self.train_loader, self.test_loader = self._get_loaders()
        self.device = torch.device("cpu")  # Use CPU for testing to avoid device mismatch
        self.model = YOLO().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = YOLOLoss()
        self.losses: dict[str, list[float | tuple[int, float]]] = {
            "coord": [],
            "obj": [],
            "noobj": [],
            "class": [],
            "loss": [],
            "map_train": [],
            "map_test": [],
        }

    def _get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Obtains the train and test loaders.

        Returns:
            Train and test loader.
        """

        train_dataset = RaccoonDataset(
            img_dir="data/images_resized/train",
            label_dir="data/labels_resized/train"
        )
        test_dataset = RaccoonDataset(
            img_dir="data/images_resized/test",
            label_dir="data/labels_resized/test"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        return train_loader, test_loader

    def _append_losses(
        self,
        coord_loss: torch.Tensor,
        obj_loss: torch.Tensor,
        noobj_loss: torch.Tensor,
        class_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Appends the losses of the batch to the dictionary of the losses.

        Args:
            coord_loss: Loss for the predicted coordinates.
            obj_loss: Loss for the prediction of if an object exists in the cell.
            noobj_loss: Loss for the prediction of if an object does not exist in the
                cell.
            class_loss: Loss for the prediction of the class of the object.
            loss: Total loss.
        """

        self.losses["coord"].append(float(coord_loss))
        self.losses["obj"].append(float(obj_loss))
        self.losses["noobj"].append(float(noobj_loss))
        self.losses["class"].append(float(class_loss))
        self.losses["loss"].append(float(loss))

    def _calculate_map(self, dataloader: DataLoader) -> float:
        """
        Calculate mean Average Precision (mAP) for a given dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            
        Returns:
            mAP value
        """
        if len(dataloader) == 10:
            subset = "train"
        else:
            subset = "test"
            
        # Use the existing utility function that reads from files
        return mean_average_precision(subset, self.model)

    def _make_epoch_train(self) -> None:
        """
        Performs one epoch of the training.
        """

        self.model.train()
        
        for imgs, targets in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(imgs)
            
            # Calculate losses
            coord_loss, obj_loss, noobj_loss, class_loss, total_loss = self.criterion(
                predictions, targets
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Store losses
            self._append_losses(coord_loss, obj_loss, noobj_loss, class_loss, total_loss)

    def fit(
        self,
        n_epochs: int = EPOCHS,
        start_epochs: list[int] | None = None,
        save_every: int = 5,
        path_weights: str = PATH_MODEL,
        path_images: str | None = None,
    ) -> None:
        """
        Training of the network. It saves the parameters of the trained model.

        Args:
            n_epochs: Number of epochs to train.
            start_epochs: List with the starting epoch for each graphic saved.
            save_every: To save the model and calculate mAP every 'save_every' epochs.
            path_weights: Path where the weights are saved.
            path_images: Path where we save the images of the evolution of the training.
        """

        # Create directories if they don't exist
        import os
        os.makedirs(os.path.dirname(path_weights), exist_ok=True)
        if path_images:
            os.makedirs(os.path.dirname(path_images), exist_ok=True)
        
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            
            # Train for one epoch
            self._make_epoch_train()
            
            # Calculate mAP every save_every epochs or at the end
            if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
                print("Calculating mAP...")
                
                # Calculate mAP for train set
                train_map = self._calculate_map(self.train_loader)
                self.losses["map_train"].append((epoch + 1, train_map))
                
                # Calculate mAP for test set
                test_map = self._calculate_map(self.test_loader)
                self.losses["map_test"].append((epoch + 1, test_map))
                
                print(f"Train mAP: {train_map:.4f}, Test mAP: {test_map:.4f}")
                
                # Save model weights
                torch.save(self.model.state_dict(), path_weights)
                print(f"Model saved to {path_weights}")
        
        # Plot training evolution
        if path_images:
            self._plot_training_evolution(start_epochs, path_images)


    def test(self) -> float:
        """
        Test of the trained network.

        Returns:
            Loss in the test set.
        """

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for imgs, targets in self.test_loader:
                # Move data to device
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(imgs)
                
                # Calculate loss
                _, _, _, _, loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save_graphic_loss(self, start_epoch: int, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            start_epoch: First epoch of the graphic.
            path_images: Path where the images are saved.
        """

        x = range(start_epoch + 1, len(self.losses["loss"]) + 1)

        fig, axs = plt.subplots()
        axs.plot(x, self.losses["coord"][start_epoch:], label="Coordinates")
        axs.plot(x, self.losses["obj"][start_epoch:], label="Objects")
        axs.plot(x, self.losses["noobj"][start_epoch:], label="No Objects")
        axs.plot(x, self.losses["class"][start_epoch:], label="Classification")
        axs.plot(x, self.losses["loss"][start_epoch:], label="Total")

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Loss")
        axs.set_title("Evolution of the Loss During the Training")

        fig.savefig(f"{path_images}_loss_{start_epoch}.png")
        plt.close(fig)

    def _save_graphic_map(self, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            path_images: Path where the images are saved.
        """

        x = [xx[0] for xx in self.losses["map_train"]]  # type: ignore

        fig, axs = plt.subplots()
        axs.plot(
            x, [xx[1] for xx in self.losses["map_train"]], label="Train"  # type: ignore
        )
        axs.plot(
            x, [xx[1] for xx in self.losses["map_test"]], label="Test"  # type: ignore
        )

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("mAP")
        axs.set_title("Evolution of the mAP During the Training")

        fig.savefig(f"{path_images}_map.png")
        plt.close(fig)

    def _plot_training_evolution(
        self, start_epochs: list[int] | None = None, path_images: str | None = None
    ) -> None:
        """
        Plots the evolution of the training loss.

        Args:
            start_epochs: List with the starting epoch for each graphic saved.
            path_images: Path where the images are saved.
        """

        if start_epochs is None:
            total_epochs = len(self.losses["loss"])
            if total_epochs < 50:
                start_epochs = [0]
            elif total_epochs < 100:
                start_epochs = [0, 50]
            else:
                start_epochs = [0, 50, 100]
        if path_images is None:
            path_images = "images/training/evolution"
            os.makedirs(path_images, exist_ok=True)

        for start_epoch in start_epochs:
            self._save_graphic_loss(start_epoch, path_images=path_images)

        self._save_graphic_map(path_images)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()
