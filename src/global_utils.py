import logging

import rasterio 
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import numpy as np
import lightning as L
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only
from skimage.measure import block_reduce
from math import gcd
import geopandas as gpd
import pooch
import json

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
            
            if "nan" in resampling_method:
                resampling_method = "nan" + resampling_method

            # We reduce in block pixel to reach the target resolution
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