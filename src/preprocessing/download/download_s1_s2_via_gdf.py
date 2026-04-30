import geefetch
import geefetch.utils
import geefetch.utils.gee
import geopandas as gpd
import hydra
from omegaconf import DictConfig
from pandas.io.formats.style import save_to_buffer
from retry import retry
import logging
from shapely.ops import unary_union
from pathlib import Path
from tqdm import tqdm
from src.preprocessing.download.download_s1_s2_utils import download_s1_s2
from omegaconf import OmegaConf
import numpy as np

@hydra.main(version_base=None, config_path="../../../configs/preprocessing/download", config_name="dwd_gdf_height_map_timeseries")
@retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:    

    print(OmegaConf.to_yaml(cfg, resolve=True))

    gdf_path = Path(cfg.gdf_path).resolve()
    data_dir = Path(cfg.data_dir).resolve()
    
    initial_gdf = gpd.read_file(gdf_path)

    save_gdf_path = data_dir / gdf_path.name
    initial_gdf.to_file(save_gdf_path, driver="GeoJSON")

    # To simplify the download process and reduce the number of images, 
    # each date is rounded to the 15th day of its respective month.
    initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str).str[:6] + "15"

    # initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates]
    
    # Group the geometries by the grouping dates
    grouped_by_df = (
        initial_gdf
        .groupby("grouping_dates")
        .agg({ 'geometry': lambda x: unary_union(x) })
        .reset_index()
    )
    grouped_gdf = gpd.GeoDataFrame(grouped_by_df, geometry='geometry', crs=initial_gdf.crs)

    grouped_gdf_filename = gdf_path.stem + "_grouped.geojson"
    grouped_gdf_path = data_dir / grouped_gdf_filename
    grouped_gdf.to_file(grouped_gdf_path, driver="GeoJSON")

    total_rows = len(grouped_gdf)
    logging.info("Starting sequential download.")
    for i, row_gdf in tqdm(grouped_gdf.iterrows(), total=total_rows, desc="Downloading data"):
        process_geometry(row_gdf, cfg, i, total_rows)
    logging.info("Download process completed.")


def process_geometry(row_gdf, cfg, i, total_rows):
    if isinstance(cfg.gee_project, str):
        cfg.gee_project = [cfg.gee_project]

    date = row_gdf["grouping_dates"]
    geometry = row_gdf["geometry"]
    
    # We ensure that each spatially disjointed polygon is processed on a sepatly of the gdf, to avoid downloading data between the two that we're not interested in.
    if geometry.geom_type == 'MultiPolygon':
            polygons = [polygon for polygon in geometry.geoms]
    else :
        polygons = [geometry]

    # Checkpoint to avoid downloading data that has already been downloaded
    ckpt_path = Path(cfg.data_dir).resolve() / "ckpt" / f"lidar_{date}.npy"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.start_from_scratch:
        ckpt_path.unlink(missing_ok=True)
    
    # Initialize ckpt
    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
    else:
        ckpt = np.empty((0, 5), dtype=np.float64)
        np.save(ckpt_path, ckpt)
    
    logging.info(f"########## date: {date}, {i}/{total_rows}")

    # Download data for each polygon
    for polygon in tqdm(polygons, total=len(polygons), desc=f"Downloading polygons associated with lidar_{date}, {i}/{total_rows}") :
        bounds = polygon.bounds # (minx, miny, maxx, maxy)

        # Create current entry
        current_entry = np.array(list(bounds) + [float(date)], dtype=np.float64)
        
        # Check if already processed
        is_processed = False
        if ckpt.shape[0] > 0:
            # Exact match check
            is_processed = np.any(np.all(ckpt == current_entry, axis=1))
            if is_processed:
                logging.info(f"Polygon {bounds} with date {date} already downloaded (skipping)")
                continue
            
        # Download data if not already processed
        download_s1_s2(
            data_dir=Path(cfg.data_dir).resolve(),
            ee_project_ids=list(cfg.gee_project),
            resolution=cfg.resolution,
            tile_shape=cfg.tile_shape,
            max_tile_size=cfg.max_tile_size,
            cloudless_portion=cfg.cloudless_portion,
            cloud_prb=cfg.cloud_prb,
            country=cfg.country,
            composite_method_s1=cfg.composite_method_s1,
            composite_method_s2=cfg.composite_method_s2,
            bounds=bounds,
            reference_date=date,
            duration=cfg.duration,
            s1_orbit=cfg.s1_orbit
        )

        # Update checkpoint
        # Reload to handle parallel updates (simple file lock mechanism would be better but this is rudimentary)
        ckpt = np.load(ckpt_path)
        ckpt = np.vstack([ckpt, current_entry])
        np.save(ckpt_path, ckpt)

if __name__ == "__main__":
    main()


