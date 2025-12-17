import numpy as np
from pathlib import Path
from src.global_utils import get_window
import torch

class getS1S2Composites:
    def __init__(
        self,
        inputs_path, 
        resolution_input, 
        date_column,
        resampling_method,
        open_even_oob,
    ):
        self.inputs_path = Path(inputs_path).resolve()
        self.resolution_input = resolution_input
        self.date_column = date_column
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.mean = None
        self.std = None

    def prepare_gdf_for_inputs(self, gdf) : 
        return gdf #No preparation needed for composite image
    
    def __len__(self) :
        return len(self.gdf)
    
    def __call__(self, bounds, row, transform):
        lidar_date = row[self.date_column]

        # Load inputs
        inputs = []

        file_lidar_date = lidar_date[:6] + "15" # files are stored with the date of the middle of the month
        s2_vrt = self.inputs_path / f"lidar_date_{file_lidar_date}" / "s2" / "s2.vrt"
        s1_asc_vrt = self.inputs_path / f"lidar_date_{file_lidar_date}" / "s1"  / "s1_asc.vrt"
        s1_dsc_vrt = self.inputs_path / f"lidar_date_{file_lidar_date}" / "s1" / "s1_dsc.vrt"
        
        s2_image, _ = get_window(
            image_path=s2_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method = self.resampling_method,
            open_even_oob = self.open_even_oob,
        )
        s1_asc_image, _ = get_window(
            image_path=s1_asc_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method = self.resampling_method,
            open_even_oob = self.open_even_oob,
        )
        s1_dsc_image, _ = get_window(
            image_path=s1_dsc_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method = self.resampling_method,
            open_even_oob = self.open_even_oob,
        )

        inputs = np.concatenate((s2_image, s1_asc_image, s1_dsc_image), axis=0)
        inputs = inputs.astype(np.float32).transpose(1, 2, 0)
        inputs[~np.isfinite(inputs)] = 0

        if transform:
            inputs = transform(inputs)

        return inputs, {}