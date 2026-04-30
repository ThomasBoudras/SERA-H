import logging

import rasterio 
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import numpy as np
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only
from skimage.measure import block_reduce
from math import gcd
import geopandas as gpd
import pooch
from tqdm import tqdm
from shapely.geometry import box
import warnings

logger = logging.getLogger(__name__)

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def print_config(
    config: DictConfig,
    fields = (
        "trainer",
        "module",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml", theme='ansiwhite'))

    rich.print(tree)

    with open("config_sera.txt", "w") as fp:
        rich.print(tree, file=fp)


def get_window(
    image_path,
    bounds,
    resolution,
    resampling_method,
    open_even_oob = False,
) : 
    """Retrieve a window from an image, within given bounds or within the bounds of a geometry"""
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()

        init_resolution = round(profile["transform"].a, 1)
        real_width, real_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        src_bounds = src.bounds
        bounds_within_vrt = (
            bounds[0] >= src_bounds.left and  
            bounds[1] >= src_bounds.bottom and 
            bounds[2] <= src_bounds.right and 
            bounds[3] <= src_bounds.top  
        )
        
        if not bounds_within_vrt and not open_even_oob:
            return None, None
        
        window = from_bounds(*bounds, transform=src.transform)
        transform = src.window_transform(window)
        
        # case resolution different from initial resolution
        if resolution is not None  and init_resolution != resolution :
            
            # case support by rasterio
            if resampling_method in {"bilinear", "cubic", "cubic_spline", "lanczos", "nearest"} :
                window_height, window_width = real_height/resolution, real_width/resolution
                resampling = getattr(Resampling, resampling_method)
            
            # case not support by rasterio
            else :
                # We are going to start from the original image, and after loading the data, 
                # we will multiply each pixel to create subpixels that can be grouped into blocks 
                # to reach the target resolution.
                window_height = np.ceil(real_height/init_resolution) #ceil to avoid rounding issues and get all needed pixels, crop is done later
                window_width = np.ceil(real_width/init_resolution) #ceil to avoid rounding issues and get all needed pixels, crop is done later
                resampling = Resampling.nearest 
            
            # Update transform to match the new resolution
            transform = Affine(
                resolution,
                transform.b,
                transform.c,
                transform.d,
                -resolution,
                transform.f,
            )
        else :
            window_height, window_width = window.height, window.width
            resampling = Resampling.nearest
       
        data = src.read(
            out_shape=(
                src.count,
                int(np.round(window_height)),
                int(np.round(window_width)),
                ),
            window=window,
            resampling=resampling,
            boundless=True,
            fill_value=np.nan
            ).astype(np.float32)
            
        if "nodata" in profile and profile["nodata"] is not None and not np.isnan(profile["nodata"]):
            data[data == profile["nodata"]] = np.nan

        # come back of the case not supported by rasterio
        if resolution is not None and resolution != init_resolution and resampling_method not in {"bilinear", "cubic", "cubic_spline", "lanczos", "nearest"} :
            if resampling_method is None:
                resampling_method = "max"
            
            # We assume the resolutions are in meters and with a precision of 1 decimeter (e.g., 1.5m). 
            # we also assume that the real height is divisible by the resolution.
            # We compute the greatest common divisor (GCD) of the two resolutions.            
            common_divisor = gcd(int(resolution*10), int(init_resolution*10))/10 # we search the commum divisor in dm
            mul_factor = int(init_resolution/common_divisor) # factor to multiply the data to create subpixels
            div_factor = int(resolution/common_divisor) # factor to divide the data to reach the target resolution

            # We repeat the data by the mul_factor to create subpixels 
            data = np.repeat(np.repeat(data, mul_factor, axis=-2), mul_factor, axis=-1)

            # we crop to be sure to have the good number of pixels 
            tmp_height = int(round(real_height/resolution))*div_factor
            tmp_width = int(round(real_width/resolution))*div_factor
            data = data[...,:tmp_height,:tmp_width] 
            
            if (
                not resampling_method.startswith("nan")
                and resampling_method in {"mean", "max", "min", "median"}
            ):
                resampling_method = "nan" + resampling_method

            # We reduce in block pixel to reach the target resolution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                data = block_reduce(data, block_size=(1, div_factor, div_factor), cval=np.nan, func=getattr(np, resampling_method))

        count = data.shape[0] if len(data.shape) == 3 else 1
        height = data.shape[-2]
        width = data.shape[-1]

        profile.update(
            {
                "transform": transform,
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": count,
                "nodata": np.nan,
                "dtype": data.dtype,
            }
        )
    return data, profile
 

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
    border_margin_size = (border_margin_size//resolution + 1) * resolution # we add the remainder to the border margin size to ensure that the patches are aligned to the resolution
    assert patch_size % resolution == 0, f"The patch size must be divisible by the resolution: {patch_size} % {resolution} != 0"

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
            # We compute the bounds of the current patch
            x_stop = x_start + patch_size
            y_stop = y_start + patch_size


            # We ensure that the last patches are within the global bounds and keep the same size as the other patches.
            x_start = min(x_start, max_x - patch_size)
            y_start = min(y_start, max_y - patch_size)
            x_stop = min(x_stop, max_x)
            y_stop = min(y_stop, max_y)
            
            # Save patch bounds
            patch_bounds.append((x_start, y_start, x_stop, y_stop))

            # Move y_start for next patch, with overlap and alignment to resolution
            if y_stop < max_y :
                y_start = y_stop - 2*border_margin_size
            else:
                break  # End of y loop

        # Move x_start for next patch, with overlap and alignment to resolution
        if x_stop < max_x :
            x_start = x_stop - 2*border_margin_size
        else:
            break  # End of x loop

    return patch_bounds

def subdivide_bounds_gdf(gdf, patch_size, margin_size, resolution) :
    """
    
    """
    new_gdf = []
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Subdividing gdf"):
        bounds = row['geometry'].bounds
        list_bounds = get_grid( global_bounds=bounds, patch_size=patch_size, margin_size=margin_size, resolution=resolution)
        global_bounds = (
            min([sub_bounds[0] for sub_bounds in list_bounds]),
            min([sub_bounds[1] for sub_bounds in list_bounds]),
            max([sub_bounds[2] for sub_bounds in list_bounds]),
            max([sub_bounds[3] for sub_bounds in list_bounds]),
        )
        for sub_bounds in list_bounds :
            new_row = row.copy()
            new_row["geometry"] = box(*sub_bounds)
            new_row["patch_bounds"] = global_bounds
            new_gdf.append(new_row)
    return gpd.GeoDataFrame(new_gdf, crs=gdf.crs).reset_index()

def expand_bounds_gdf(gdf, margin_size, resolution) :
    """
    Expand the bounds of the gdf to include all the subpatches.
    """
    new_gdf = []

    border_margin_size = int(np.ceil(1.5 * margin_size))
    border_margin_size = (border_margin_size//resolution + 1) * resolution # we add the remainder to the border margin size to ensure that the patches are aligned to the resolution
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Expanding bounds gdf"):
        bounds = row['geometry'].bounds
        bounds = (
            bounds[0] - border_margin_size,
            bounds[1] - border_margin_size,
            bounds[2] + border_margin_size,
            bounds[3] + border_margin_size,
        )
        new_row = row.copy()
        new_row["geometry"] = box(*bounds)
        new_gdf.append(new_row)
    return gpd.GeoDataFrame(new_gdf, crs=gdf.crs).reset_index()

class DiffGdfAdapter   :
    def __init__(self, country="France", min_images=1, date_columns=["date"]):
        """
        Args:
            country (str): The country name to check inclusion (default: "France").
            min_images (int): Minimal number of images required per geometry.
            date_columns (list[str]): Name of the columns containing image dates/info.
        """
        self.country = country
        self.min_images = min_images
        self.date_columns = date_columns
        country_borders_url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "world-administrative-boundaries/exports/geojson"
        )
        country_borders_path = pooch.retrieve(url=country_borders_url, known_hash=None)
        country_borders = gpd.read_file(country_borders_path)
        if self.country not in country_borders.name.values:
            raise ValueError(f"Unknown country {self.country}.")
        self.country_geo = country_borders[country_borders.name == self.country]

    def __call__(self, gdf):
        """
        Subdivides gdf geometries into patches, and filters by: image count and intersection in the country.

        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            patch_size (float): Size of each patch.
            resolution (float, optional): Resolution for get_grid (defaults to patch_size).

        Returns:
            gpd.GeoDataFrame: Adapted GeoDataFrame.
        """
        
        # Filter by number of images for each date column
        for col in self.date_columns:
            if col in gdf.columns:
                gdf = gdf[gdf[col].apply(lambda x: len(x)) >= self.min_images]
    
        # Reproject country shape to match gdf's CRS for accurate intersection
        country_geo_reprojected = self.country_geo.to_crs(gdf.crs)
        country_shape_reprojected = country_geo_reprojected.geometry.unary_union
        
        gdf = gdf[gdf["geometry"].intersects(country_shape_reprojected)]

        return gdf.reset_index(drop=True)