"""Utility sampler and helper functions shared by the datamodule and datasets."""

from typing import Iterator, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SubsetSampler(Sampler):
    """Sampler that draws indices from a fixed-size subset of a dataset.

    Used to limit the number of samples seen per epoch on reduced datasets.
    """

    def __init__(self, data_source: Dataset, num_samples: int, shuffle: bool = True) -> None:
        """Initializes the sampler.

        Args:
            data_source: Dataset to sample indices from.
            num_samples: Requested number of samples; capped to `len(data_source)`.
            shuffle: Whether to shuffle the indices at each iteration.
        """
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the sampled indices."""
        indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of samples drawn per epoch."""
        return self.num_samples


def get_random_tensor_crop(
    input_shape: Union[Tuple[int, ...], torch.Tensor], patch_size: int
) -> Tuple[int, int, int, int]:
    """Computes a random square crop location inside an image.

    Args:
        input_shape: Shape of the input tensor (as a tuple or a tensor of sizes);
            the last two dimensions are interpreted as (height, width).
        patch_size: Side length of the square crop.

    Returns:
        A tuple `(x_start, y_start, x_stop, y_stop)` of pixel indices delimiting the crop.

    Raises:
        ValueError: If `patch_size` is larger than the image height or width.
    """
    height, width = input_shape[-2:]
    if height < patch_size or width < patch_size:
        raise ValueError("patch_size must be smaller than both height and width of the image.")

    # Randomly select the top and left coordinates of the crop
    max_x_start = height - patch_size
    max_y_start = width - patch_size
    x_start = np.random.randint(0, max_x_start + 1)
    y_start = np.random.randint(0, max_y_start + 1)

    # Compute bottom and right indices
    x_stop = x_start + patch_size
    y_stop = y_start + patch_size

    return (x_start, y_start, x_stop, y_stop)
