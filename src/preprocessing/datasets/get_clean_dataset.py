import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from shapely.geometry import box
from src.global_utils import subdivide_bounds_gdf
import math

@hydra.main(version_base=None, config_path="../../../configs/preprocessing/datasets", config_name="get_clean_dataset")
def main(cfg: DictConfig) -> None:    
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    create_vrts = hydra.utils.get_method(cfg.create_vrts)
    get_valid_vrts = hydra.utils.get_method(cfg.get_valid_vrts)
    data_dir = Path(cfg.data_dir).resolve()

    if cfg.get("via_bounds", None) is not None:
        row = {name : data for name, data in cfg.via_bounds.items() if name!="bounds"}
        bounds = cfg.via_bounds["bounds"]
        bounds = (
            int(math.floor(bounds["left"] / 1000) * 1000),
            int(math.floor(bounds["bottom"] / 1000) * 1000),
            int(math.ceil(bounds["right"] / 1000) * 1000),
            int(math.ceil(bounds["top"] / 1000) * 1000),
        )

        row["geometry"] = box(*bounds)
        aoi_gdf = gpd.GeoDataFrame([row], crs="EPSG:2154")

        initial_gdf = subdivide_bounds_gdf(aoi_gdf, patch_size=cfg.via_bounds["patch_size"], margin_size=0, resolution=cfg.via_bounds["resolution"])
        initial_gdf[cfg.grouping_dates_column] = cfg.via_bounds["date"].replace("/", "")
    
    elif cfg.get("via_geojson", None) is not None:
        path = cfg.via_geojson["path"]
        initial_gdf = gpd.read_file(path)
        if cfg.via_geojson.get("date", None) is not None:
            initial_gdf[cfg.grouping_dates_column] = cfg.via_geojson["date"].replace("/", "")
        
    else:
        raise ValueError("Either 'via_geojson' or 'via_bounds' must be provided, but not both.")
    
    logging.info(f"The size of the gdf is : {len(initial_gdf)}")
    

    create_vrts(data_dir)
    
    # Find valid VRTs for each geometry
    logging.info("Starting to compute 'vrt_list_timeseries' with parallel processing.")
    tqdm_desc = "Find correct VRT for each geometries"
    initial_gdf[cfg.validation_column] = Parallel(n_jobs=cfg.n_jobs_parrallelized)(
        delayed(get_valid_vrts)(data_dir, row["geometry"], row[cfg.grouping_dates_column], cfg.half_window_size) 
        for _, row in tqdm(initial_gdf.iterrows(), total=len(initial_gdf), desc=tqdm_desc)
    )
    
    tqdm.pandas(desc="Delete bounds without data")
    logging.info("Filtering out rows with no valid vrt'.")
    clean_gdf = initial_gdf[initial_gdf[cfg.validation_column].notnull()].reset_index(drop=True).copy()
    unclean_gdf = initial_gdf[initial_gdf[cfg.validation_column].isnull()].reset_index(drop=True).copy()
    
    clean_gdf.to_file(cfg.gdf_clean_path, driver="GeoJSON")
    unclean_gdf.to_file(cfg.gdf_unclean_path, driver="GeoJSON")
    logging.info(f"Cleaned geodataframe saved to {cfg.gdf_clean_path}. Process complete.")


if __name__ == "__main__":
    main()




