import numpy as np
from src.global_utils  import get_window
import torch
from pathlib import Path

class getSpotComposites:
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

        spot_vrt = self.inputs_path / f"{lidar_date[:4]}" / "spot.vrt"

        inputs, _ = get_window(
            image_path=spot_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method = self.resampling_method,
            open_even_oob = self.open_even_oob,
            )
        
        inputs = inputs.astype(np.float32).transpose(1, 2, 0)
        inputs[~np.isfinite(inputs)] = 0

        if transform:
            inputs = transform(inputs)

        return inputs, {}