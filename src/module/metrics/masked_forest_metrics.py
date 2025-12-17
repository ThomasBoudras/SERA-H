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
        forest_mask_path,
        classification_path,
        classes_to_keep,
        resolution_target,
    ):
        super().__init__()
        self.metrics_calculator = metrics_calculator
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path) if forest_mask_path is not None else None
        self.classification_path = Path(classification_path).resolve() if classification_path is not None else None
        self.classes_to_keep = classes_to_keep
        self.resolution_target = resolution_target

        for name, reduce_fx in self.metrics_calculator.get_required_states().items():
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx=reduce_fx)


    # Update the metrics for each batch
    def update(self, pred, target, meta_data):
        # Get the vegetation mask
        if self.classification_path is not None:
            bounds_batch = meta_data["bounds"]
            if "lidar_years" in meta_data:
                lidar_years_batch = meta_data["lidar_years"]
            else : 
                lidar_years_batch = meta_data["classification_years"] #change map dataset case (we have juste one year where we have classification)
            vegetation_mask = []
            for i, bounds in enumerate(bounds_batch):
                year = lidar_years_batch[i].item()
                classification_path = self.classification_path / str(year) / "lidar_classification.vrt"
                mask, _ = get_vegetation_and_forest_mask(
                    forest_mask_gdf=self.forest_mask_gdf,
                    classification_raster_path=classification_path,
                    bounds=bounds.tolist(),
                    classes_to_keep=self.classes_to_keep,
                    resolution=self.resolution_target,
                    resampling_method="nearest", #we load a mask, nearest is the best for this case
                )
                mask = torch.from_numpy(np.expand_dims(mask, axis=0)).to(target.device)
                vegetation_mask.append(mask)
        else : 
            # If no classification path, create a mask full of true values
            vegetation_mask = [torch.ones_like(pred[0], dtype=torch.bool) for _ in range(len(pred))]
        
        # Stack the vegetation masks and create a mask for the nan values
        vegetation_mask = torch.stack(vegetation_mask).to(target.device)
        nan_mask = torch.isnan(target)
        mask = vegetation_mask & ~nan_mask

        # Update the metrics
        self.metrics_calculator.batch_update(
            pred = pred.to(target.device), 
            target = target.to(target.device), 
            mask = mask, 
            states = self, 
        )

    # Compute the final results
    def compute(self):
        final_results = self.metrics_calculator.epoch_compute(states=self)
        return final_results





