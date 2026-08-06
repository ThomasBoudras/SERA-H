"""Target loader returning NaN placeholders, used when no ground-truth is available."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch


class getNan:
    """Target loader that returns NaN placeholders instead of real height targets.

    Used for prediction stages where no ground-truth height data is available.
    """

    def __init__(self, date_column: Optional[str]) -> None:
        """Initializes the placeholder target loader.

        Args:
            date_column: Name of the GeoDataFrame column holding the acquisition
                date, or None if not available.
        """
        self.date_column = date_column

    def __call__(
        self, bounds: List[float], row: pd.Series, transform: Optional[Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns a NaN target tensor for the sample.

        Args:
            bounds: Spatial bounds of the patch (unused, kept for interface parity).
            row: GeoDataFrame row for the sample.
            transform: Unused, kept for interface parity with other target loaders.

        Returns:
            A tuple `(targets, metadata)` where `targets` is a scalar NaN tensor, and
            `metadata` contains the acquisition date if `date_column` is set.
        """
        if self.date_column is not None:
            lidar_date = int(row[self.date_column])
            return torch.tensor(torch.nan), {"lidar_acquisition_date": torch.tensor(lidar_date)}
        return torch.tensor(torch.nan), {}
