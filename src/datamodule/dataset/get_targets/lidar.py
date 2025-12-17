import numpy as np
import torch
from pathlib import Path
from src.global_utils  import get_window


class getLidarImages :
    def __init__(
        self, 
        targets_path, 
        resolution_target, 
        replace_nan_by_zero_in_target, 
        date_column, 
        unit_column,
        resampling_method,
    ):
        
        self.targets_path = Path(targets_path).resolve()
        self.resolution_target = resolution_target
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.date_column = date_column
        self.unit_column = unit_column
        self.resampling_method = resampling_method
        
        self.scaling_factor = {"m": 1, "dm": 10, "cm": 100, "mm": 1000}


    def __call__(self, bounds, row, transform) :
        lidar_date = row[self.date_column]
        unit = row[self.unit_column]

        lidar_vrt  = self.targets_path / f"{lidar_date[:4]}" / "lidar.vrt"

        targets, _ = get_window(
            image_path=lidar_vrt,
            bounds=bounds,
            resolution=self.resolution_target,
            resampling_method = self.resampling_method
        )
        targets = targets.astype(np.float32).transpose(1, 2, 0)

        if self.replace_nan_by_zero_in_target:
            targets[~np.isfinite(targets)] = 0

        targets = targets / self.scaling_factor[unit]

        if transform:
            targets = transform(targets)
        else :
            targets = torch.from_numpy(targets)

        return targets, {"lidar_years": torch.tensor(int(lidar_date[:4]))}