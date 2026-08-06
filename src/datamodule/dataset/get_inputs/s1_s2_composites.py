"""Input loader building a single-date Sentinel-1/Sentinel-2 composite input."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from src.global_utils import get_window


class getS1S2Composites:
    """Builds a Sentinel-1/Sentinel-2 composite input for a sample.

    Unlike `getS1S2Timeseries`, this loader reads a single static composite image
    per sensor (no time dimension).
    """

    def __init__(
        self,
        inputs_path: str,
        resolution_input: float,
        date_column: str,
        resampling_method: str,
        open_even_oob: bool,
    ) -> None:
        """Initializes the composite input loader.

        Args:
            inputs_path: Root directory containing the Sentinel-1/Sentinel-2 composites.
            resolution_input: Spatial resolution (in meters) to resample inputs to.
            date_column: Name of the GeoDataFrame column holding the reference date.
            resampling_method: Resampling method used when reading image windows.
            open_even_oob: Whether to read windows extending outside the raster bounds.
        """
        self.inputs_path = Path(inputs_path).resolve()
        self.resolution_input = resolution_input
        self.date_column = date_column
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.mean = None
        self.std = None

    def prepare_gdf_for_inputs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Returns the GeoDataFrame unchanged (no filtering needed for composites).

        Args:
            gdf: GeoDataFrame of samples.

        Returns:
            The same GeoDataFrame, unmodified.
        """
        return gdf  # No preparation needed for composite image

    def __len__(self) -> int:
        """Returns the number of samples in `self.gdf`."""
        return len(self.gdf)

    def __call__(
        self, bounds: List[float], row: pd.Series, transform: Optional[Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Builds the composite input tensor for a single sample.

        Loads the S2, S1 ascending, and S1 descending composite windows, concatenates
        them along the channel axis, and applies the given transform if provided.

        Args:
            bounds: Spatial bounds `(minx, miny, maxx, maxy)` of the patch to read.
            row: GeoDataFrame row for the sample, providing `date_column`.
            transform: Optional transform applied to the concatenated image.

        Returns:
            A tuple `(inputs, metadata)` where `inputs` is the composite input tensor
            and `metadata` is an empty dict.
        """
        # Load inputs
        inputs = []

        s2_vrt = self.inputs_path / "s2" / "s2.vrt"
        s1_asc_vrt = self.inputs_path / "s1" / "s1_asc.vrt"
        s1_dsc_vrt = self.inputs_path / "s1" / "s1_dsc.vrt"

        s2_image, _ = get_window(
            image_path=s2_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob,
        )
        s1_asc_image, _ = get_window(
            image_path=s1_asc_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob,
        )
        s1_dsc_image, _ = get_window(
            image_path=s1_dsc_vrt,
            bounds=bounds,
            resolution=self.resolution_input,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob,
        )

        inputs = np.concatenate((s2_image, s1_asc_image, s1_dsc_image), axis=0)
        inputs = inputs.astype(np.float32).transpose(1, 2, 0)
        inputs[~np.isfinite(inputs)] = 0

        if transform:
            inputs = transform(inputs)

        return inputs, {}
