"""Utilities to build a combined vegetation/forest mask from IGN forest polygons and
LidarHD point-cloud classification rasters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box

from src.global_utils import get_window


def get_vegetation_and_forest_mask(
    ign_forest_mask_gdf: Optional[gpd.GeoDataFrame],
    lidarhd_classification_raster_path: Union[str, Path],
    bounds: List[float],
    classes_to_keep: List[int],
    resolution: float,
    resampling_method: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Builds a boolean vegetation/forest mask over a given window.

    The mask is the union of two sources: pixels classified as vegetation in the LidarHD
    classification raster (per `classes_to_keep`), and pixels falling inside an IGN forest
    polygon (if `ign_forest_mask_gdf` is provided).

    Args:
        ign_forest_mask_gdf: GeoDataFrame of IGN forest polygons, or None to skip this source
            (in which case the corresponding mask is all False).
        lidarhd_classification_raster_path: Path to the LidarHD classification raster (or VRT)
            to read the window from.
        bounds: Spatial bounds `(minx, miny, maxx, maxy)` of the window to extract.
        classes_to_keep: LidarHD classification codes considered as vegetation.
        resolution: Target spatial resolution (in the raster's CRS units) for the window.
        resampling_method: Resampling method used when reading the window (see `get_window`).

    Returns:
        A tuple `(final_mask, lidarhd_profile)` where `final_mask` is a boolean array of the
        window's shape (True where the pixel is vegetation and/or forest), and
        `lidarhd_profile` is the rasterio profile of the read window.
    """
    raster_bounds = box(*bounds)

    # mask from classification of lidarhd data
    lidarhd_classification, lidarhd_profile = get_window(
        lidarhd_classification_raster_path,
        bounds=bounds,
        resolution=resolution,
        resampling_method=resampling_method,
        open_even_oob=False,
    )
    lidarhd_classification = lidarhd_classification.squeeze()
    lidarhd_classif_mask = lidarhd_classification == classes_to_keep[0]
    if len(classes_to_keep) > 1:
        for aclass in classes_to_keep[1::]:
            lidarhd_classif_mask = lidarhd_classif_mask | (lidarhd_classification == aclass)

    # mask from forest mask of ONF
    if ign_forest_mask_gdf is not None:
        clipped_gdf = gpd.clip(ign_forest_mask_gdf, raster_bounds)
        geometries = [(geom, 1) for geom in clipped_gdf.geometry]
        if len(geometries):
            ign_forest_mask = rasterize(
                geometries,
                out_shape=lidarhd_classification.shape,
                transform=lidarhd_profile["transform"],
                fill=0,
                default_value=1,
                dtype=np.uint8,
            ).astype(bool)
        else:
            ign_forest_mask = np.zeros_like(lidarhd_classif_mask, dtype=bool)
    else:
        ign_forest_mask = np.zeros_like(lidarhd_classif_mask, dtype=bool)

    # Combine the two masks
    final_mask = lidarhd_classif_mask | ign_forest_mask
    return final_mask, lidarhd_profile
