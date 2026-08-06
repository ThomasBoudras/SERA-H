"""Utilities used by the ``LogImages`` callback: TensorBoard logger lookup and helper
classes to prepare and log input/target/prediction images.
"""

from typing import Any, List, Tuple

import matplotlib as mpl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.utils import make_grid


# Get tensorboard logger
def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Retrieves the TensorBoard logger from a Lightning trainer.

    Args:
        trainer (Trainer): The Lightning trainer, either using a single logger or a
            list of loggers.

    Returns:
        TensorBoardLogger: The TensorBoard logger attached to the trainer.

    Raises:
        Exception: If no ``TensorBoardLogger`` is found among the trainer's logger(s).
    """

    # If the logger is already a TensorBoardLogger, return it
    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    # If trainer.logger is a list, look for a TensorBoardLogger in the list
    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    raise Exception(
        "You are using a TensorBoard related callback, but TensorBoardLogger was not found for some reason..."
    )


# Prepare images functions
class height_map_mode:
    """Class for preparing and coloring 'height map' images.

    Args:
        min_value_normalization (float): Minimum value for normalization.
        max_value_normalization (float): Maximum value for normalization.
        colormap (str): Name of the matplotlib colormap to use (e.g., 'magma', 'viridis').
    """

    def __init__(
        self, min_value_normalization: float, max_value_normalization: float, colormap: str
    ) -> None:
        self.min_value_normalization = min_value_normalization
        self.max_value_normalization = max_value_normalization
        self.colormap = colormap

    def __call__(
        self, pred_image: torch.Tensor, target_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes and applies the colormap to a prediction/target pair.

        Args:
            pred_image (torch.Tensor): Predicted height map tensor.
            target_image (torch.Tensor): Target height map tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The colored prediction and target images.
        """
        pred_image, target_image = self._normalize_pred_and_target(
            pred_tensor=pred_image, target_tensor=target_image
        )
        target_image = self._color_transform(target_image)
        pred_image = self._color_transform(pred_image)
        return pred_image, target_image

    def _normalize_pred_and_target(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes prediction and target tensors to the [0, 1] range.

        Args:
            pred_tensor (torch.Tensor): Predicted height map tensor.
            target_tensor (torch.Tensor): Target height map tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized and clamped prediction
            and target tensors.
        """
        normalized_pred = (pred_tensor - self.min_value_normalization) / (
            self.max_value_normalization - self.min_value_normalization
        )
        normalized_target = (target_tensor - self.min_value_normalization) / (
            self.max_value_normalization - self.min_value_normalization
        )
        # Clamp to ensure values are between 0 and 1
        normalized_pred = normalized_pred.clamp(0, 1)
        normalized_target = normalized_target.clamp(0, 1)
        return normalized_pred, normalized_target

    def _color_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the matplotlib colormap to a single-channel tensor.

        Args:
            tensor (torch.Tensor): Single-channel tensor of shape (1, H, W) or (H, W).

        Returns:
            torch.Tensor: RGB tensor of shape (3, H, W) on the same device as `tensor`.
        """
        colormap = mpl.colormaps[self.colormap]
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor_np = tensor.squeeze(0).cpu().numpy()
        else:
            tensor_np = tensor.cpu().numpy()
        colored = colormap(tensor_np)  # (H, W, 4)
        colored = colored[..., :3]  # remove alpha
        colored = torch.from_numpy(colored).permute(2, 0, 1).float().to(device=tensor.device)
        return colored


# Log input images functions
class log_4d_s1_s2_images:
    """Class for logging 4D sentinel-1/sentinel-2 images (num_samples, channels, height, width) to tensorboard.

    Args:
        None
    """

    def __call__(self, inputs: List[torch.Tensor], experiment: Any, stage: str) -> None:
        """Logs a grid of 4D input images to TensorBoard.

        Args:
            inputs (List[torch.Tensor]): List of input image tensors to stack and log.
            experiment (Any): TensorBoard ``SummaryWriter``-like experiment object.
            stage (str): Name of the stage used to namespace the logged images.
        """
        input_images = torch.stack(inputs, dim=0)
        input_images = input_images[:, [2, 1, 0], :, :]  # Convert BGR to RGB
        input_images = make_grid(input_images, normalize=True)
        # Log image to tensorboard
        experiment.add_image(f"input_images/{stage}", input_images, global_step=0)


class log_5d_s1_s2_images:
    """Class for logging 5D sentinel-1/sentinel-2 images (num_samples, time, channels, height, width) to tensorboard.

    Args:
        max_nb_timeseries_input (int): Maximum number of time steps to log.
    """

    def __init__(self, max_nb_timeseries_input: int) -> None:
        self.max_nb_timeseries_input = max_nb_timeseries_input

    def __call__(self, inputs: List[torch.Tensor], experiment: Any, stage: str) -> None:
        """Logs a median composite and per-timestep grids of 5D input images to TensorBoard.

        Args:
            inputs (List[torch.Tensor]): List of input image tensors to stack and log.
            experiment (Any): TensorBoard ``SummaryWriter``-like experiment object.
            stage (str): Name of the stage used to namespace the logged images.
        """
        input_images = torch.stack(inputs, dim=0)
        input_images = input_images[:, :, [2, 1, 0], :, :]  # Convert BGR to RGB
        median_inputs = input_images.median(dim=1).values
        median_inputs = make_grid(median_inputs, normalize=True)
        experiment.add_image(
            f"input_images/{stage}/median", median_inputs, global_step=0
        )  # As the input are the same each epoch, we dont specify the epoch

        for t in range(min(input_images.shape[1], self.max_nb_timeseries_input)):
            input_images_t = input_images[:, t, :, :, :]
            input_images_t = make_grid(input_images_t, normalize=True)
            experiment.add_image(
                f"input_images/{stage}/time_{t}", input_images_t, global_step=0
            )  # As the input are the same each epoch, we dont specify the epoch
