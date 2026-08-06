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
        area_column,
        unit_column,
        resampling_method,
    ):
        
        self.targets_path = targets_path
        self.resolution_target = resolution_target
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.date_column = date_column
        self.area_column = area_column
        self.unit_column = unit_column
        self.resampling_method = resampling_method
        
        self.scaling_factor = {"m": 1, "dm": 10, "cm": 100, "mm": 1000}

    def __call__(self, bounds, row, transform) :
        lidar_date = row[self.date_column] if self.date_column is not None else None
        area = row[self.area_column] if self.area_column is not None else None
        unit = row[self.unit_column] if self.unit_column is not None and self.unit_column in row else "m"
 
        if lidar_date is not None:
            lidar_vrt  = self.targets_path.replace("<lidar_date>", lidar_date[:4])
        if area is not None:
            lidar_vrt  = lidar_vrt.replace("<area>", area)

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

        return targets, {"lidar_acquisition_date": torch.tensor([int(lidar_date)])}