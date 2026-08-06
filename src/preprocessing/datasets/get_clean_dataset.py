import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
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

    # Find valid VRTs for each geometry.
    # Processed by chunks: after each chunk the results are checkpointed to disk and
    # the loky worker pool is shut down, which both frees the workers' resident memory
    # (GDAL caches, etc.) and allows resuming after a crash/OOM.
    logging.info("Starting to compute 'vrt_list_timeseries' with parallel processing.")
    tqdm_desc = "Find correct VRT for each geometries"

    chunk_size = cfg.get("chunk_size", 2000)
    checkpoint_path = Path(cfg.get("checkpoint_path", None) or (str(Path(cfg.gdf_clean_path).with_suffix("")) + "_checkpoint.pkl"))

    # results_dict maps a gdf index -> result (a list of valid vrts, or None if no data)
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results_dict = pickle.load(f)
        logging.info(f"Resuming from checkpoint {checkpoint_path}: {len(results_dict)}/{len(initial_gdf)} geometries already processed.")
    else:
        results_dict = {}

    todo = [idx for idx in initial_gdf.index if idx not in results_dict]
    logging.info(f"{len(todo)} geometries left to process (chunk_size={chunk_size}).")

    for start in range(0, len(todo), chunk_size):
        chunk_idx = todo[start:start + chunk_size]
        chunk_results = Parallel(n_jobs=cfg.n_jobs_parrallelized)(
            delayed(get_valid_vrts)(
                data_dir,
                initial_gdf.at[idx, "geometry"],
                initial_gdf.at[idx, cfg.grouping_dates_column],
                cfg.half_window_size,
            )
            for idx in tqdm(chunk_idx, desc=tqdm_desc, total=len(chunk_idx))
        )
        for idx, res in zip(chunk_idx, chunk_results):
            results_dict[idx] = res

        # Atomic checkpoint write so an interrupt mid-write can't corrupt the file.
        tmp_path = checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(results_dict, f)
        tmp_path.replace(checkpoint_path)
        logging.info(f"Checkpoint saved: {len(results_dict)}/{len(initial_gdf)} geometries processed.")

        # Tear down the worker pool to release its resident memory before the next chunk.
        get_reusable_executor().shutdown(wait=True)

    initial_gdf[cfg.validation_column] = initial_gdf.index.map(results_dict)

    tqdm.pandas(desc="Delete bounds without data")
    logging.info("Filtering out rows with no valid vrt'.")
    clean_gdf = initial_gdf[initial_gdf[cfg.validation_column].notnull()].reset_index(drop=True).copy()
    unclean_gdf = initial_gdf[initial_gdf[cfg.validation_column].isnull()].reset_index(drop=True).copy()
    
    clean_gdf.to_file(cfg.gdf_clean_path, driver="GeoJSON")
    unclean_gdf.to_file(cfg.gdf_unclean_path, driver="GeoJSON")
    logging.info(f"Cleaned geodataframe saved to {cfg.gdf_clean_path}. Process complete.")

    # Everything is written: the checkpoint is no longer needed.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logging.info(f"Checkpoint {checkpoint_path} removed.")


if __name__ == "__main__":
    main()




