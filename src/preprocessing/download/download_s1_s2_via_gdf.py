"""Download Sentinel-1/Sentinel-2 tiles for a geodataframe of areas of interest.

Reads a geodataframe of areas of interest, groups their geometries by a rounded acquisition
date, and sequentially downloads the corresponding S1/S2 imagery for each group via
`download_s1_s2`, checkpointing progress with marker files so an interrupted run can resume.
"""

import logging
from pathlib import Path

import geopandas as gpd
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from retry import retry
from shapely.ops import unary_union
from tqdm import tqdm

from src.preprocessing.download.download_s1_s2_utils import download_s1_s2


@hydra.main(
    version_base=None,
    config_path="../../../configs/preprocessing/download",
    config_name="dwd_gdf_height_map_timeseries",
)
@retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:
    """Group the areas of interest by date and sequentially download S1/S2 imagery for each group.

    The input geodataframe (`cfg.gdf_path`) is copied to `cfg.data_dir`, then its geometries are
    grouped by a rounded acquisition date (the 15th of the month, unless all dates are already
    identical) and merged with `unary_union`. Each group is downloaded via `process_dates`, with
    the whole function retried up to 100 times (10s delay) on any exception.

    Args:
        cfg: Hydra configuration with the input `gdf_path`, the output `data_dir`, the
            `grouping_dates` column name, and the download parameters forwarded to
            `download_s1_s2`.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    gdf_path = Path(cfg.gdf_path).resolve()
    data_dir = Path(cfg.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    initial_gdf = gpd.read_file(gdf_path)

    save_gdf_path = data_dir / gdf_path.name
    initial_gdf.to_file(save_gdf_path, driver="GeoJSON")

    # To simplify the download process and reduce the number of images,
    # each date is rounded to the 15th day of its respective month.
    initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str)
    if not initial_gdf["grouping_dates"].nunique() == 1:
        # Gather dates that are close within a month under the 15th of that month.
        initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str).str[:6] + "15"

    # Group the geometries by the grouping dates
    grouped_by_df = (
        initial_gdf.groupby("grouping_dates")
        .agg({"geometry": lambda x: unary_union(x)})
        .reset_index()
    )
    grouped_gdf = gpd.GeoDataFrame(grouped_by_df, geometry="geometry", crs=initial_gdf.crs)

    grouped_gdf_path = data_dir / (gdf_path.stem + "_grouped_by_dates.geojson")
    grouped_gdf.to_file(grouped_gdf_path, driver="GeoJSON")

    total_rows = len(grouped_gdf)
    logging.info("Starting sequential download.")
    for i, row_gdf in tqdm(grouped_gdf.iterrows(), total=total_rows, desc="Downloading data"):
        process_dates(row_gdf, cfg, i, total_rows)
    logging.info("Download process completed.")


def process_dates(row_gdf: pd.Series, cfg: DictConfig, i: int, total_rows: int) -> None:
    """Download S1/S2 imagery for a single grouped date, skipping it if already downloaded.

    A marker file (`<data_dir>/ckpt/<date>.done`) is used to checkpoint completed dates: it is
    only created after `download_s1_s2` returns successfully, so an interrupted download never
    leaves a false "done" state behind.

    Args:
        row_gdf: Row of the grouped geodataframe, with a "grouping_dates" and a "geometry" field.
        cfg: Hydra configuration with the download parameters forwarded to `download_s1_s2`.
        i: Index of this row among the rows being processed (used for logging).
        total_rows: Total number of rows being processed (used for logging).
    """
    if isinstance(cfg.gee_project, str):
        cfg.gee_project = [cfg.gee_project]

    date = row_gdf["grouping_dates"]
    geometry = row_gdf["geometry"]

    bounds_global = geometry.bounds

    # Checkpoint: one marker file per date. A date is considered downloaded once its marker
    # exists, which only happens after download_s1_s2 returns successfully (see marker.touch()
    # below), so a failed/interrupted download never leaves behind a false "done" state.
    ckpt_dir = Path(cfg.data_dir).resolve() / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    marker = ckpt_dir / f"{date}.done"

    if cfg.get("start_from_scratch", False):
        marker.unlink(missing_ok=True)

    if marker.exists():
        logging.info(f"Date {date} already downloaded (skipping) [{i}/{total_rows}]")
        return

    logging.info(f"########## date: {date}, {i}/{total_rows} ##########")

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
        bounds=bounds_global,
        reference_date=date,
        duration=cfg.duration,
        s1_orbit=cfg.s1_orbit,
        filter_polygon=geometry,
    )

    marker.touch()


if __name__ == "__main__":
    main()
