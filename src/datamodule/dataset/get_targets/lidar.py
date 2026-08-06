"""Target loader reading LiDAR-derived canopy height images."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.global_utils import get_window


class getLidarImages:
    """Builds the LiDAR canopy height target for a sample.

    Reads the target image window for the appropriate lidar acquisition date/area,
    converts it to the requested unit, and optionally replaces NaNs with zeros.
    """

    def __init__(
        self,
        targets_path: str,
        resolution_target: float,
        replace_nan_by_zero_in_target: bool,
        date_column: Optional[str],
        area_column: Optional[str],
        unit_column: Optional[str],
        resampling_method: str,
    ) -> None:
        """Initializes the LiDAR target loader.

        Args:
            targets_path: Path template to the LiDAR raster, with `<lidar_date>` and
                `<area>` placeholders substituted per sample.
            resolution_target: Spatial resolution (in meters) to resample targets to.
            replace_nan_by_zero_in_target: Whether to replace non-finite target
                values with zero.
            date_column: Name of the GeoDataFrame column holding the lidar
                acquisition date, or None if not used to build the target path.
            area_column: Name of the GeoDataFrame column holding the area name, or
                None if not used to build the target path.
            unit_column: Name of the GeoDataFrame column holding the height unit
                ("m", "dm", "cm", or "mm"), or None to default to meters.
            resampling_method: Resampling method used when reading the target window.
        """

        self.targets_path = targets_path
        self.resolution_target = resolution_target
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target
        self.date_column = date_column
        self.area_column = area_column
        self.unit_column = unit_column
        self.resampling_method = resampling_method

        self.scaling_factor = {"m": 1, "dm": 10, "cm": 100, "mm": 1000}

    def __call__(
        self, bounds: List[float], row: pd.Series, transform: Optional[Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Builds the LiDAR height target tensor for a single sample.

        Args:
            bounds: Spatial bounds `(minx, miny, maxx, maxy)` of the patch to read.
            row: GeoDataFrame row for the sample, providing `date_column`,
                `area_column`, and `unit_column` as configured.
            transform: Optional transform applied to the target array.

        Returns:
            A tuple `(targets, metadata)` where `targets` is the height target
            (converted to meters), and `metadata` holds the lidar acquisition date.
        """
        lidar_date = row[self.date_column] if self.date_column is not None else None
        area = row[self.area_column] if self.area_column is not None else None
        unit = (
            row[self.unit_column]
            if self.unit_column is not None and self.unit_column in row
            else "m"
        )

        if lidar_date is not None:
            lidar_vrt = self.targets_path.replace("<lidar_date>", lidar_date[:4])
        if area is not None:
            lidar_vrt = lidar_vrt.replace("<area>", area)

        targets, _ = get_window(
            image_path=lidar_vrt,
            bounds=bounds,
            resolution=self.resolution_target,
            resampling_method=self.resampling_method,
        )
        targets = targets.astype(np.float32).transpose(1, 2, 0)

        if self.replace_nan_by_zero_in_target:
            targets[~np.isfinite(targets)] = 0

        targets = targets / self.scaling_factor[unit]

        if transform:
            targets = transform(targets)
        else:
            targets = torch.from_numpy(targets)

        return targets, {"lidar_acquisition_date": torch.tensor([int(lidar_date)])}
