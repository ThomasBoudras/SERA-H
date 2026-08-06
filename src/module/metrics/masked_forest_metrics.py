import torch
import numpy as np
import geopandas as gpd
from torchmetrics import Metric
from pathlib import Path

from src.module.metrics.masked_forest_metrics_utils import get_vegetation_and_forest_mask

class maskedForestMetrics(Metric): 
    # Initialize the metrics
    def __init__(
        self,
        metrics_calculator,
        ign_forest_mask_path,
        lidarhd_classification_mask_path,
        classes_to_keep,
        resolution_target,
    ):
        super().__init__()
        self.metrics_calculator = metrics_calculator
        self.ign_forest_mask_gdf = gpd.read_parquet(ign_forest_mask_path) if ign_forest_mask_path is not None else None
        self.lidarhd_classification_mask_path = Path(lidarhd_classification_mask_path).resolve() if lidarhd_classification_mask_path is not None else None
        self.classes_to_keep = classes_to_keep
        self.resolution_target = resolution_target

        for name, reduce_fx in self.metrics_calculator.get_required_states().items():
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx=reduce_fx)


    # Update the metrics for each batch
    def update(self, pred, target, metadata):
        # Get the vegetation mask
        if self.lidarhd_classification_mask_path is not None:
            bounds_batch = metadata["bounds"]
            if "lidar_acquisition_date" in metadata:
                lidar_years_batch = [str(date.item())[:4] for date in metadata["lidar_acquisition_date"]]
            else : 
                lidar_years_batch = metadata["classification_years"] #change map dataset case (we have juste one year where we have classification)
            vegetation_mask = []
            for i, bounds in enumerate(bounds_batch):
                year = lidar_years_batch[i]
                year = year.item() if hasattr(year, "item") else year  # tensor (classification_years) vs str (lidar_acquisition_date)
                lidarhd_classification_mask_path = self.lidarhd_classification_mask_path / str(year) / "lidar_classification.vrt"
                mask, _ = get_vegetation_and_forest_mask(
                    ign_forest_mask_gdf=self.ign_forest_mask_gdf,
                    lidarhd_classification_raster_path=lidarhd_classification_mask_path,
                    bounds=bounds.tolist(),
                    classes_to_keep=self.classes_to_keep,
                    resolution=self.resolution_target,
                    resampling_method="nearest", #we load a mask, nearest is the best for this case
                )
                mask = torch.from_numpy(np.expand_dims(mask, axis=0)).to(target.device)
                vegetation_mask.append(mask)
        else : 
            # If no classification path, create a mask full of true values
            vegetation_mask = [torch.ones_like(target[0], dtype=torch.bool) for _ in range(len(target))]
        
        # Stack the vegetation masks and create a mask for the nan values
        vegetation_mask = torch.stack(vegetation_mask).to(target.device)
        nan_mask = torch.isnan(target)
        mask = vegetation_mask & ~nan_mask

        # Update the metrics
        self.metrics_calculator.batch_update(
            pred = pred.to(target.device), 
            target = target, 
            mask = mask, 
            states = self, 
        )

    # Compute the final results
    def compute(self):
        final_results = self.metrics_calculator.epoch_compute(states=self)
        return final_results





