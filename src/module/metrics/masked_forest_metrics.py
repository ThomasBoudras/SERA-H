"""Torchmetrics `Metric` that restricts height-map metrics computation to forested,
vegetated pixels using IGN forest and LidarHD classification masks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import torch
from torchmetrics import Metric

from src.module.metrics.masked_forest_metrics_utils import get_vegetation_and_forest_mask


class maskedForestMetrics(Metric):
    """Wraps a metrics calculator to only accumulate statistics on forest/vegetation pixels.

    Combines an IGN forest mask and a LidarHD classification-based vegetation mask with the
    NaN mask of the target, then delegates the actual per-batch state update and epoch-level
    computation to `metrics_calculator`.
    """

    # Initialize the metrics
    def __init__(
        self,
        metrics_calculator: Any,
        ign_forest_mask_path: Optional[str],
        lidarhd_classification_mask_path: Optional[str],
        classes_to_keep: List[int],
        resolution_target: float,
    ) -> None:
        """Initializes the metric and its accumulator states.

        Args:
            metrics_calculator: Object exposing `get_required_states`, `batch_update` and
                `epoch_compute`, used to accumulate and compute the actual metrics
                (e.g. `heightMapMetrics`).
            ign_forest_mask_path: Path to a GeoParquet file with IGN forest polygons, or
                None to skip the forest mask (vegetation mask will be all False in that case).
            lidarhd_classification_mask_path: Path to the directory of LidarHD classification
                rasters (organized by year), or None to skip the vegetation mask entirely
                (in which case all pixels are considered valid, subject to the NaN mask).
            classes_to_keep: LidarHD classification codes considered as vegetation.
            resolution_target: Target spatial resolution (in meters) used when loading the
                classification raster window.
        """
        super().__init__()
        self.metrics_calculator = metrics_calculator
        self.ign_forest_mask_gdf = (
            gpd.read_parquet(ign_forest_mask_path) if ign_forest_mask_path is not None else None
        )
        self.lidarhd_classification_mask_path = (
            Path(lidarhd_classification_mask_path).resolve()
            if lidarhd_classification_mask_path is not None
            else None
        )
        self.classes_to_keep = classes_to_keep
        self.resolution_target = resolution_target

        for name, reduce_fx in self.metrics_calculator.get_required_states().items():
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx=reduce_fx)

    # Update the metrics for each batch
    def update(self, pred: torch.Tensor, target: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """Updates the accumulator states with one batch of predictions and targets.

        Args:
            pred: Predicted tensor, same shape as `target`.
            target: Target tensor, may contain NaN values which are excluded from the mask.
            metadata: Batch metadata dict. When `lidarhd_classification_mask_path` is set,
                must contain "bounds" (per-sample spatial bounds) and either
                "lidar_acquisition_date" or "classification_years" to locate the matching
                classification raster for each sample.
        """
        # Get the vegetation mask
        if self.lidarhd_classification_mask_path is not None:
            bounds_batch = metadata["bounds"]
            if "lidar_acquisition_date" in metadata:
                lidar_years_batch = [
                    str(date.item())[:4] for date in metadata["lidar_acquisition_date"]
                ]
            else:
                lidar_years_batch = metadata[
                    "classification_years"
                ]  # change map dataset case (we have juste one year where we have classification)
            vegetation_mask = []
            for i, bounds in enumerate(bounds_batch):
                year = lidar_years_batch[i]
                year = (
                    year.item() if hasattr(year, "item") else year
                )  # tensor (classification_years) vs str (lidar_acquisition_date)
                lidarhd_classification_mask_path = (
                    self.lidarhd_classification_mask_path / str(year) / "lidar_classification.vrt"
                )
                mask, _ = get_vegetation_and_forest_mask(
                    ign_forest_mask_gdf=self.ign_forest_mask_gdf,
                    lidarhd_classification_raster_path=lidarhd_classification_mask_path,
                    bounds=bounds.tolist(),
                    classes_to_keep=self.classes_to_keep,
                    resolution=self.resolution_target,
                    resampling_method="nearest",  # we load a mask, nearest is the best for this case
                )
                mask = torch.from_numpy(np.expand_dims(mask, axis=0)).to(target.device)
                vegetation_mask.append(mask)
        else:
            # If no classification path, create a mask full of true values
            vegetation_mask = [
                torch.ones_like(target[0], dtype=torch.bool) for _ in range(len(target))
            ]

        # Stack the vegetation masks and create a mask for the nan values
        vegetation_mask = torch.stack(vegetation_mask).to(target.device)
        nan_mask = torch.isnan(target)
        mask = vegetation_mask & ~nan_mask

        # Update the metrics
        self.metrics_calculator.batch_update(
            pred=pred.to(target.device),
            target=target,
            mask=mask,
            states=self,
        )

    # Compute the final results
    def compute(self) -> Dict[str, torch.Tensor]:
        """Computes the final metric values over all accumulated states.

        Returns:
            Dict mapping metric names to their computed tensor values, as returned by
            `self.metrics_calculator.epoch_compute`.
        """
        final_results = self.metrics_calculator.epoch_compute(states=self)
        return final_results
