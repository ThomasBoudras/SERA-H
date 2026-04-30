import numpy as np
from torch.utils.data import Sampler
from shapely.geometry import box
import geopandas as gpd
from tqdm import tqdm

class SubsetSampler(Sampler):
    """
    Subset sampler in the case of reduced dataset.
    """
    def __init__(self, data_source, num_samples, shuffle=True):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_random_tensor_crop(input_shape, patch_size):
    """
    Returns a random n x n square crop from inside an image of shape (height, width).
    Returns (top, left, bottom, right) pixel indices for the crop.
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

