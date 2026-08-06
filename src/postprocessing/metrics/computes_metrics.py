"""Hydra entry point that orchestrates the full metrics postprocessing pipeline.

Loads the AOI geometries, dispatches per-row image loading/metrics/local plotting
across parallel workers, then aggregates local results into global metrics and
global plots, optionally saving results to Excel.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import global_utils as utils


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/postprocessing/metrics",
    config_name="height_predictions.yaml",
)
def compute_metrics(config: DictConfig) -> None:
    """Runs the metrics postprocessing pipeline defined by the Hydra config.

    Args:
        config: Hydra configuration describing the geometries to process, the
            image loaders, local/global metrics computers and plotters to
            instantiate, and where to save the outputs.
    """

    if config.get("print_config"):
        utils.print_config(
            config,
            resolve=True,
            fields={"get_images", "get_metrics_local", "get_metrics_global", "get_plots"},
        )
    OmegaConf.set_struct(config, True)

    if "seed" in config:
        seed_everything(config.seed)

    # Read the geometries and save them in save_dir
    aoi_gdf = gpd.read_file(config.geometries_path)

    if "test" in config.version_metrics:
        aoi_gdf = aoi_gdf[aoi_gdf["split"] == "test"].reset_index(drop=True)

    aoi_gdf_path = Path(config.save_dir) / "gdf_metrics.geojson"
    if config.get("nb_samples", None) is not None:
        aoi_gdf = aoi_gdf.sample(config.nb_samples).reset_index(drop=True)
    aoi_gdf.to_file(aoi_gdf_path, driver="GeoJSON")

    # Split the GeoDataFrame into chunks for parallel processing.
    nb_jobs = config.get("nb_jobs")
    if config.get("chunk_size", None) is not None:
        chunk_size = config.chunk_size
    else:
        chunk_size = len(aoi_gdf) // (nb_jobs * 4)
        chunk_size = chunk_size if chunk_size > 0 else 1
    aoi_gdf_chunks = [aoi_gdf.iloc[i : i + chunk_size] for i in range(0, len(aoi_gdf), chunk_size)]

    # Instantiate the plots local if needed
    if config.get("get_plots_local", None) is not None:
        nb_plots = int(min(len(aoi_gdf), config.get_plots_local.nb_plots))
        indice_plot_local = np.random.choice(len(aoi_gdf), nb_plots, replace=False)
        save_dir_local = config.get_plots_local.save_dir
        if Path(save_dir_local).exists():
            shutil.rmtree(save_dir_local)
        Path(save_dir_local).mkdir(parents=True)

    else:
        indice_plot_local = None

    # Instantiate the plots global if needed
    if config.get("get_plots_global", None) is not None:
        save_dir_global = config.get_plots_global.save_dir
        if Path(save_dir_global).exists():
            shutil.rmtree(save_dir_global)
        Path(save_dir_global).mkdir(parents=True)

    def process_chunk(
        config: DictConfig, gdf_chunk: gpd.GeoDataFrame, indice_plot: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Processes a chunk of the GeoDataFrame.

        Args:
            config: Hydra configuration used to instantiate the image loader and
                the local metrics/plots computers.
            gdf_chunk: Subset of the AOI GeoDataFrame to process.
            indice_plot: Row indices for which a local plot should be generated,
                or None if no local plotting is requested.

        Returns:
            The list of local metrics dictionaries computed for each row of the chunk.
        """

        # To avoid bottleneck, we instantiate the get_images here, each job will use its own instance
        get_images = hydra.utils.instantiate(config.get_images)
        get_metrics_local = hydra.utils.instantiate(config.get_metrics_local)
        get_plots_local = hydra.utils.instantiate(config.get_plots_local)

        metrics_list = []
        for idx_row, row in gdf_chunk.iterrows():
            try:
                bounds = row["geometry"].bounds
                # Load images only when needed for metrics computation or plotting
                if get_metrics_local is not None or (
                    (get_plots_local is not None) and (idx_row in indice_plot_local)
                ):
                    images = get_images(row=row)
                # Compute metrics for this row if needed
                if get_metrics_local is not None:
                    metrics_local = get_metrics_local(images=images, row=row)
                    metrics_list.append(metrics_local)
                else:
                    metrics_local = {}
                # Plot for this row if needed
                if (get_plots_local is not None) and (idx_row in indice_plot):
                    plot_name_sample = f"{int(bounds[0])}_{int(bounds[1])}"
                    get_plots_local(
                        images, metrics_local, plot_name_sample=plot_name_sample, row=row
                    )
            except Exception as e:
                logging.error(f"Error processing row {idx_row}: {e}")
                continue
        return metrics_list

    # Process chunks in parallel
    results_in_chunks = Parallel(n_jobs=nb_jobs)(
        delayed(process_chunk)(
            config,
            chunk,
            indice_plot_local,
        )
        for chunk in tqdm(aoi_gdf_chunks, total=len(aoi_gdf_chunks), desc="Processing chunks")
    )

    # Flatten the list of lists of metric dicts into a single list of metric dicts
    metrics_local_list = [item for sublist in results_in_chunks for item in sublist]

    # Compute global metrics
    get_metrics_global = hydra.utils.instantiate(config.get_metrics_global)
    if get_metrics_global is not None:

        # Aggregate local results
        metrics_local_aggregated = {}
        for metrics_local in metrics_local_list:
            for name_metrics, value_metrics in metrics_local.items():
                if name_metrics not in metrics_local_aggregated:
                    metrics_local_aggregated[name_metrics] = [value_metrics]
                else:
                    metrics_local_aggregated[name_metrics].append(value_metrics)

        metrics_global = get_metrics_global(metrics_local_aggregated)

        if get_metrics_global.metrics_to_save == "all":
            metrics_to_save = metrics_global
        elif get_metrics_global.metrics_to_save is not None:
            metrics_to_save = {}
            for metric in get_metrics_global.metrics_to_save:
                metrics_to_save[metric] = metrics_global[metric]
        else:
            metrics_to_save = {}

        if metrics_to_save:
            df = pd.DataFrame(list(metrics_to_save.items()), columns=["Metrics", "Value"])
            output_path_xlsx = Path(get_metrics_global.output_path_xlsx).resolve()
            if output_path_xlsx.exists():
                with pd.ExcelWriter(
                    output_path_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace"
                ) as writer:
                    df.to_excel(writer, index=False, sheet_name=config.version_metrics)
            else:
                df.to_excel(output_path_xlsx, index=False, sheet_name=config.version_metrics)
    else:
        metrics_global = {}

    # Plot global metrics if needed
    get_plots_global = hydra.utils.instantiate(config.get_plots_global)
    if get_plots_global is not None:
        metrics_global_plots = get_plots_global(metrics_global)

        if get_plots_global.metrics_to_save == "all":
            global_plots_metrics_to_save = metrics_global_plots
        elif get_plots_global.metrics_to_save is not None:
            global_plots_metrics_to_save = {}
            for metric in get_plots_global.metrics_to_save:
                global_plots_metrics_to_save[metric] = metrics_global_plots[metric]
        else:
            global_plots_metrics_to_save = {}

        if global_plots_metrics_to_save:
            df = pd.DataFrame(
                list(global_plots_metrics_to_save.items()), columns=["Metrics", "Value"]
            )
            output_path_xlsx = Path(get_plots_global.output_path_xlsx).resolve()
            if output_path_xlsx.exists():
                with pd.ExcelWriter(
                    output_path_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace"
                ) as writer:
                    df.to_excel(writer, index=False, sheet_name=config.version_metrics)
            else:
                df.to_excel(output_path_xlsx, index=False, sheet_name=config.version_metrics)


if __name__ == "__main__":
    compute_metrics()
