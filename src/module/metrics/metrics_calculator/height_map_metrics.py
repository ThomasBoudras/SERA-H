"""Height-map metrics calculator (error, relative error, and tree-cover IoU metrics),
designed to be accumulated pixel-wise across an epoch rather than averaged per image.
"""

from typing import Any, Dict

import torch


class heightMapMetrics:
    """Computes pixel-level error and tree-cover metrics accumulated over a full epoch.

    Unlike per-image averaging, this calculator accumulates sums (error, squared error,
    absolute error, relative error, tree-cover intersection/union, pixel counts) across all
    batches of an epoch so that the final metrics reflect a global, pixel-wise average.
    """

    # For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, tree_cover_threshold: float, min_height_nMAE: float) -> None:
        """Initializes the calculator.

        Args:
            tree_cover_threshold: Height threshold (in meters) above which a pixel is
                considered tree cover, used for the tree-cover IoU metric.
            min_height_nMAE: Minimum target height (in meters) for a pixel to be included in
                the normalized MAE (nMAE) computation.
        """
        self.tree_cover_threshold = tree_cover_threshold
        self.min_height_nMAE = min_height_nMAE

    def get_required_states(self) -> Dict[str, str]:
        """Describes the accumulator states this calculator needs and how to reduce them in DDP.

        Returns:
            Dict mapping state name to the `dist_reduce_fx` reduction to use when the state
            is registered via `Metric.add_state` (all states use "sum" here).
        """
        # Define the "contract": the states this calculator needs, and for each one,
        # how it should be reduced in DDP.
        return {
            "sum_error": "sum",
            "sum_squared_error": "sum",
            "sum_absolute_error": "sum",
            "sum_relative_error": "sum",
            "intersection": "sum",
            "union": "sum",
            "nb_values": "sum",
            "nb_values_min_height": "sum",
        }

    def batch_update(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, states: Any
    ) -> None:
        """Accumulates error and tree-cover statistics for one batch into `states`.

        Args:
            pred: Predicted tensor, same shape as `target`.
            target: Target tensor, same shape as `pred`.
            mask: Boolean tensor, same shape as `target`, selecting the valid pixels to
                include in the error metrics (tree-cover metrics use the full tensors).
            states: Object holding the accumulator state tensors declared in
                `get_required_states` (e.g. `sum_error`, `sum_squared_error`, ...), updated
                in place.
        """
        # Apply mask to target and prediction
        masked_target = target[mask]
        masked_pred = pred[mask]

        if len(masked_pred) > 0:
            # Compute error metrics
            error = masked_pred - masked_target
            states.sum_error += torch.sum(error)
            states.sum_squared_error += torch.sum(error**2)
            states.sum_absolute_error += torch.sum(torch.abs(error))

            # Compute relative error metrics
            mask_min_height = masked_target >= self.min_height_nMAE
            states.sum_relative_error += torch.sum(
                torch.abs(masked_pred[mask_min_height] - masked_target[mask_min_height])
                / (1 + masked_target[mask_min_height])
            )

            # Compute tree cover metrics
            tree_cover_target = target >= self.tree_cover_threshold
            tree_cover_pred = pred >= self.tree_cover_threshold
            states.intersection += torch.logical_and(tree_cover_pred, tree_cover_target).sum()
            states.union += torch.logical_or(tree_cover_pred, tree_cover_target).sum()

            # Compute number of values for each mask
            states.nb_values += mask.sum()
            states.nb_values_min_height += mask_min_height.sum()

    def epoch_compute(self, states: Any) -> Dict[str, torch.Tensor]:
        """Computes the final epoch-level metrics from the accumulated states.

        Args:
            states: Object holding the accumulator state tensors declared in
                `get_required_states`, accumulated over the epoch via `batch_update`.

        Returns:
            Dict with keys "ME", "MAE", "RMSE", "nMAE", "TreeCov" and "nb_values". All
            values are NaN (and "nb_values" is 0) if `states.nb_values` is 0.
        """
        final_results = {}
        # If there are values, compute metrics
        if states.nb_values > 0:
            # ME - Mean Error
            final_results["ME"] = (states.sum_error / states.nb_values).to(torch.float32)

            # MAE - Mean Absolute Error
            final_results["MAE"] = (states.sum_absolute_error / states.nb_values).to(torch.float32)

            # RMSE - Root Mean Square Error
            final_results["RMSE"] = torch.sqrt(states.sum_squared_error / states.nb_values).to(
                torch.float32
            )

            final_results["nMAE"] = (states.sum_relative_error / states.nb_values_min_height).to(
                torch.float32
            )

            # TreeCov - Treecover IoU
            final_results["TreeCov"] = (
                (states.intersection / states.union).to(torch.float32)
                if states.union > 0
                else torch.tensor(torch.nan)
            )

            # nb_values - Number of values
            final_results["nb_values"] = states.nb_values.to(torch.float32)
        else:
            # If no values, set all metrics to NaN
            final_results["ME"] = torch.tensor(torch.nan)
            final_results["MAE"] = torch.tensor(torch.nan)
            final_results["RMSE"] = torch.tensor(torch.nan)
            final_results["nMAE"] = torch.tensor(torch.nan)
            final_results["TreeCov"] = torch.tensor(torch.nan)
            final_results["nb_values"] = torch.tensor(0)

        return final_results
