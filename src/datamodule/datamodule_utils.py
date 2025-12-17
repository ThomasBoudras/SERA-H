import numpy as np
from torch.utils.data import Sampler
from shapely.geometry import box
import geopandas as gpd
       

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


def generate_random_bounds(bounds, patch_size, resolution):
    """
    Generate random bounds for data augmentation.
    """
    bounds = list(bounds)
    available_width = bounds[2] - bounds[0] - patch_size
    available_height = bounds[3] - bounds[1] - patch_size

    x_offset = np.random.randint(0, available_width // resolution + 1)
    y_offset = np.random.randint(0, available_height // resolution + 1)

    bounds[0] += x_offset * resolution
    bounds[1] += y_offset * resolution
    bounds[2] = bounds[0] + patch_size
    bounds[3] = bounds[1] + patch_size

    return bounds


def get_grid(global_bounds, patch_size, margin_size, resolution):
    """
    Get the bounds of all subpatches for a given global bounds (include margin).

    Args:
        global_bounds (tuple): (min_x, min_y, max_x, max_y) of the area.
        patch_size (int or float): Size of each patch (in the same units as bounds).
        margin_size (int or float): Margin to add around the area (in the same units as bounds).
        resolution (int or float): Step size for alignment (in the same units as bounds).

    Returns:
        list of tuple: List of (min_x, min_y, max_x, max_y) for each patch.
    """
    patch_bounds = []

    # We set 3 times the margin size: two for the edges to reduce border effects, 
    # and a third to create an overlap between neighbouring patches. 
    # This allows the values to be averaged over the overlap area, thus reducing discontinuities between patches.
    border_margin_size = int(np.ceil(1.5 * margin_size))
    
    min_x = global_bounds[0] - border_margin_size
    min_y = global_bounds[1] - border_margin_size
    max_x = global_bounds[2] + border_margin_size
    max_y = global_bounds[3] + border_margin_size

    max_x = max(min_x + patch_size, max_x)
    max_y = max(min_y + patch_size, max_y)

    x_start = min_x
    while x_start < max_x:
        y_start = min_y
        while y_start < max_y - 2*border_margin_size:
            x_stop = x_start + patch_size
            y_stop = y_start + patch_size

            # Save patch bounds
            x_start = min(x_start, max_x - patch_size)
            y_start = min(y_start, max_y - patch_size)
            x_stop = min(x_stop, max_x)
            y_stop = min(y_stop, max_y)
            patch_bounds.append((x_start, y_start, x_stop, y_stop))

            # Move y_start for next patch, with overlap and alignment to resolution
            if y_stop < max_y :
                y_start = y_stop - 2*border_margin_size
                y_start = y_start - ((y_start - min_y) % resolution) 
            else:
                break  # End of y loop

        # Move x_start for next patch, with overlap and alignment to resolution
        if x_stop < max_x :
            x_start = x_stop - 2*border_margin_size
            x_start = x_start - ((x_start - min_x) % resolution) 
        else:
            break  # End of x loop

    return patch_bounds

def expand_gdf(gdf, patch_size, margin_size, resolution) :
    """
    Expand the gdf to include all the subpatches.
    """
    new_gdf = []
    for i, row in gdf.iterrows() :
        bounds = row['geometry'].bounds
        list_bounds = get_grid( global_bounds=bounds, patch_size=patch_size, margin_size=margin_size, resolution=resolution)
        for sub_bounds in list_bounds :
            new_row = row.copy()
            new_row["geometry"] = box(*sub_bounds)
            new_gdf.append(new_row)
    return gpd.GeoDataFrame(new_gdf, crs=gdf.crs).reset_index()
