import matplotlib.colors as mcolors
import torch
from matplotlib import cm
from torchvision.utils import make_grid
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

#Get tensorboard logger
def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Get TensorBoard logger. """
    
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

#Prepare images functions
class height_map_mode:
    """
    Class for preparing and coloring 'height map' images.

    Args:
        min_value_normalization (float): Minimum value for normalization.
        max_value_normalization (float): Maximum value for normalization.
        colormap (str): Name of the matplotlib colormap to use (e.g., 'magma', 'viridis').
    """
    def __init__(self, min_value_normalization, max_value_normalization, colormap):
        self.min_value_normalization = min_value_normalization
        self.max_value_normalization = max_value_normalization
        self.colormap = colormap

    def __call__(self, pred_image, target_image):
        pred_image, target_image = self._normalize_pred_and_target(pred_tensor=pred_image, target_tensor=target_image)
        target_image = self._color_transform(target_image)
        pred_image = self._color_transform(pred_image)
        return pred_image, target_image
    
    def _normalize_pred_and_target(self, pred_tensor, target_tensor):
        normalized_pred = (pred_tensor - self.min_value_normalization) / (self.max_value_normalization - self.min_value_normalization)
        normalized_target = (target_tensor - self.min_value_normalization) / (self.max_value_normalization - self.min_value_normalization)
        # Clamp to ensure values are between 0 and 1
        normalized_pred = normalized_pred.clamp(0, 1)
        normalized_target = normalized_target.clamp(0, 1)
        return normalized_pred, normalized_target

    def _color_transform(self, tensor):
        colormap = cm.get_cmap(self.colormap)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor_np = tensor.squeeze(0).cpu().numpy()
        else:
            tensor_np = tensor.cpu().numpy()
        colored = colormap(tensor_np)  # (H, W, 4)
        colored = colored[..., :3]  # remove alpha
        colored = torch.from_numpy(colored).permute(2, 0, 1).float().to(device=tensor.device)
        return colored

class difference_map_mode:
    """
    Class for preparing and coloring 'difference map' images.

    Args:
        min_value_normalization (float): Minimum value for normalization.
        max_value_normalization (float): Maximum value for normalization.
        colormap (str): Name of the matplotlib colormap to use.
    """
    def __init__(self, min_value_normalization, max_value_normalization, colormap):
        self.min_value_normalization = min_value_normalization
        self.max_value_normalization = max_value_normalization
        self.colormap = colormap

    def __call__(self, pred_image, target_image):
        pred_image, target_image = self._normalize_pred_and_target(pred_tensor=pred_image, target_tensor=target_image)
        target_image = self._color_transform(target_image)
        pred_image = self._color_transform(pred_image)
        return pred_image, target_image
    
    def _normalize_pred_and_target(self, pred_tensor, target_tensor):
        # For difference maps, we use TwoSlopeNorm for centered normalization
        norm = mcolors.TwoSlopeNorm(vmin=self.min_value_normalization, vcenter=0, vmax=self.max_value_normalization)
        
        # Convert to numpy for normalization
        pred_np = pred_tensor.cpu().numpy()
        target_np = target_tensor.cpu().numpy()
        
        # Apply normalization
        normalized_pred = torch.from_numpy(norm(pred_np))
        normalized_target = torch.from_numpy(norm(target_np))
        
        return normalized_pred, normalized_target

    def _color_transform(self, tensor):
        colormap = cm.get_cmap(self.colormap)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor_np = tensor.squeeze(0).cpu().numpy()
        else:
            tensor_np = tensor.cpu().numpy()
        colored = colormap(tensor_np)  # (H, W, 4)
        colored = colored[..., :3]  # remove alpha
        colored = torch.from_numpy(colored).permute(2, 0, 1).float().to(device=tensor.device)
        return colored

class change_map_mode:
    """
    Class for preparing 'change map' images.

    Args:
        None
    """
    def __call__(self, pred_image, target_image):
        target_image = target_image.float()
        return pred_image, target_image


#Log input images functions
class log_4d_s1_s2_images:
    """
    Class for logging 4D sentinel-1/sentinel-2 images (num_samples, channels, height, width) to tensorboard.

    Args:
        None
    """
    def __call__(self, inputs, experiment, stage):
        input_images = torch.stack(inputs, dim=0)
        input_images = input_images[:, [2, 1, 0], :, :]  # Convert BGR to RGB
        input_images = make_grid(input_images, normalize=True)
        # Log image to tensorboard
        experiment.add_image(f"input_images/{stage}", input_images, global_step=0)

class log_5d_s1_s2_images:
    """
    Class for logging 5D sentinel-1/sentinel-2 images (num_samples, time, channels, height, width) to tensorboard.

    Args:
        max_nb_timeseries_input (int): Maximum number of time steps to log.
    """
    def __init__(self, max_nb_timeseries_input):
        self.max_nb_timeseries_input = max_nb_timeseries_input

    def __call__(self, inputs, experiment, stage):
        input_images = torch.stack(inputs, dim=0)
        input_images = input_images[:, :, [2, 1, 0], :, :]  # Convert BGR to RGB
        median_inputs = input_images.median(dim=1).values
        median_inputs = make_grid(median_inputs, normalize=True)
        experiment.add_image(f"input_images/{stage}/median", median_inputs, global_step=0)  # As the input are the same each epoch, we dont specify the epoch

        for t in range(min(input_images.shape[1], self.max_nb_timeseries_input)):
            input_images_t = input_images[:, t, :, :, :]
            input_images_t = make_grid(input_images_t, normalize=True)
            experiment.add_image(f"input_images/{stage}/time_{t}", input_images_t, global_step=0)  # As the input are the same each epoch, we dont specify the epoch


class log_5d_s1_s2_images_revisit(log_5d_s1_s2_images):
    """
    Class for logging 5D sentinel-1/sentinel-2 images (num_samples, time, channels, height, width) to tensorboard.

    Args:
        None
    """
    def __call__(self, inputs, experiment, stage):
        input_separator = inputs[0].shape[0] // 2
        input_images_t1 = [input_i[:input_separator, :, :, :].contiguous() for input_i in inputs]
        input_images_t2 = [input_i[input_separator:, :, :, :].contiguous() for input_i in inputs]
        
        super().__call__(input_images_t1, experiment, f"{stage}_t1")
        super().__call__(input_images_t2, experiment, f"{stage}_t2")


class log_4d_spot_images:
    """
    Class for logging 4D Spot image (num_samples, channels, height, width) to tensorboard.

    Args:
        None
    """
    def __call__(self, inputs, experiment, stage):
        input_images = torch.stack(inputs, dim=0)
        input_images = input_images[:, :3, :, :]  # Keep just RGB 
        input_images = make_grid(input_images, normalize=True)
        experiment.add_image(f"input_images/{stage}", input_images, global_step=0)  # As the input are the same each epoch, we dont specify the epoch