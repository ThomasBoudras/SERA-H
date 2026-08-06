import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
import geopandas as gpd

from src.global_utils  import get_window


def get_vegetation_and_forest_mask(
    ign_forest_mask_gdf,
    lidarhd_classification_raster_path,
    bounds,
    classes_to_keep,
    resolution,
    resampling_method,
):
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
    else :
        ign_forest_mask = np.zeros_like(lidarhd_classif_mask, dtype=bool)
    
    # Combine the two masks
    final_mask = lidarhd_classif_mask | ign_forest_mask
    return final_mask, lidarhd_profile
