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
from joblib import Parallel, delayed
from tqdm import tqdm
from src.preprocessing.download.download_s1_s2_utils import download_s1_s2

@hydra.main(version_base=None, config_path="../../../configs/preprocessing/download", config_name="dwd_gdf_lidarhd_timeseries")
# @retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:    

    gdf_path = Path(cfg.gdf_path).resolve()
    data_dir = Path(cfg.data_dir).resolve()
    
    initial_gdf = gpd.read_file(gdf_path)

    save_gdf_path = data_dir / gdf_path.name
    initial_gdf.to_file(save_gdf_path, driver="GeoJSON")

    # To simplify the download process and reduce the number of images, 
    # each date is rounded to the 15th day of its respective month.
    initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str).str[:6] + "15"
    
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

    # Parallelisation 
    total_rows = len(grouped_gdf)
    logging.info("Starting parallel download.")
    Parallel(n_jobs=cfg.n_jobs_parrallelized)(
        delayed(process_geometry)(row_gdf, cfg)
        for _, row_gdf in tqdm(grouped_gdf.iterrows(), total=total_rows, desc="Downloading data")
    )
    logging.info("Download process completed.")


def process_geometry(row_gdf, cfg):
    if isinstance(cfg.gee_project, str):
        cfg.gee_project = [cfg.gee_project]

    """Process a single geometry row and perform the download."""
    for project in list(cfg.gee_project):
        geefetch.utils.gee.auth(project)
    
    date = row_gdf["grouping_dates"]
    geometry = row_gdf["geometry"]
    
    # We ensure that each spatially disjointed polygon is processed on a sepatly of the gdf, to avoid downloading data between the two that we're not interested in.
    if geometry.geom_type == 'MultiPolygon':
            polygons = [polygon for polygon in geometry.geoms]
    else :
        polygons = [geometry]

    for polygon in tqdm(polygons, desc =f"Download polygons associated with lidar_{date}",total=len(polygons)) :
        bounds = polygon.bounds
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

        

if __name__ == "__main__":
    main()


