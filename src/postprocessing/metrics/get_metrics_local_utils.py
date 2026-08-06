"""Computers that derive per-sample (local) metrics from loaded images.

Each computer implements a `compute_metrics` method taking the images of a
single sample and returns either a scalar value or a dict of named values,
which `get_metrics_local` aggregates into a single metrics dictionary.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.measure import label


class get_metrics_local:
    """Runs a set of local metric computers on a single sample's images.

    Args:
        metrics_set: Mapping from metric name to a computer object exposing a
            `compute_metrics(images, metrics, row)` method.
    """

    def __init__(self, metrics_set: Dict[str, Any]) -> None:
        self.metrics_set = metrics_set

    def __call__(self, images: Dict[str, np.ndarray], row: pd.Series) -> Dict[str, Any]:
        """Computes all configured local metrics for a sample.

        Args:
            images: Dictionary of loaded/computed images for the sample.
            row: GeoDataFrame row corresponding to the sample.

        Returns:
            Dictionary mapping metric names to their computed values. When a
            computer returns a dict, its keys are prefixed with the metric name.
        """
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metric_value = metric_computer.compute_metrics(images, metrics, row)
            if isinstance(metric_value, dict):
                for key, value in metric_value.items():
                    metrics[f"{metric_name}_{key}"] = value
            else:
                metrics[metric_name] = metric_value
        return metrics


class true_positive_object_scale_local_computer:
    """Counts true-positive connected components in an object-scale evaluation mask.

    Args:
        name_change: Base name of the change image whose evaluation mask is used.
        reduction_basis: Either "pred" or "ref", selecting which side's mask
            (`{name_change}_pred` or `{name_change}_ref`) to read.

    Raises:
        AssertionError: If `reduction_basis` is not "pred" or "ref".
    """

    def __init__(self, name_change: str, reduction_basis: str) -> None:
        self.name_change = name_change
        self.reduction_basis = reduction_basis
        assert self.reduction_basis in ["pred", "ref"], "reduction_basis must be 'pred' or 'ref'"

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> int:
        """Counts the true-positive connected components (value == 1) in the mask.

        Args:
            images: Dictionary of images, must contain `{name_change}_{reduction_basis}`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            The number of true-positive connected components.

        Raises:
            Exception: If the required image is not present in `images`.
        """
        name_metrics = f"{self.name_change}_{self.reduction_basis}"
        if name_metrics not in images:
            raise Exception(f"You must first load {name_metrics}")

        image_change = images[name_metrics]

        # In the evaluation mask, 1 represents a True Positive (in the ref channel)
        tp_mask = (image_change == 1).astype(np.uint8)
        _, num_tp = label(tp_mask, return_num=True)

        return num_tp


class false_positive_object_scale_local_computer:
    """Counts false-positive connected components in the predicted evaluation mask.

    Args:
        name_change: Base name of the change image; the mask read is
            `{name_change}_pred`.
    """

    def __init__(self, name_change: str) -> None:
        self.name_change = name_change

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> int:
        """Counts the false-positive connected components (value == -1) in the mask.

        Args:
            images: Dictionary of images, must contain `{name_change}_pred`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            The number of false-positive connected components.

        Raises:
            Exception: If the required image is not present in `images`.
        """
        name_metrics = f"{self.name_change}_pred"
        if name_metrics not in images:
            raise Exception(f"You must first load {name_metrics}")

        image_change_pred = images[name_metrics]

        # In the evaluation mask, -1 represents a False Positive (in the pred channel)
        fp_mask = (image_change_pred == -1).astype(np.uint8)
        _, num_fp = label(fp_mask, return_num=True)

        return num_fp


class false_negative_object_scale_local_computer:
    """Counts false-negative connected components in the reference evaluation mask.

    Args:
        name_change: Base name of the change image; the mask read is
            `{name_change}_ref`.
    """

    def __init__(self, name_change: str) -> None:
        self.name_change = name_change

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> int:
        """Counts the false-negative connected components (value == -1) in the mask.

        Args:
            images: Dictionary of images, must contain `{name_change}_ref`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            The number of false-negative connected components.

        Raises:
            Exception: If the required image is not present in `images`.
        """
        name_metrics = f"{self.name_change}_ref"
        if name_metrics not in images:
            raise Exception(f"You must first load {name_metrics}")

        image_change_ref = images[name_metrics]

        # In the evaluation mask, -1 represents a False Negative (in the ref channel)
        fn_mask = (image_change_ref == -1).astype(np.uint8)
        _, num_fn = label(fn_mask, return_num=True)

        return num_fn


class binned_by_object_scale_local_computer:
    """Computes an object-scale metric independently for a series of area bins.

    Args:
        name_change: Base name of the change image; the per-bin image read is
            `{name_change}_{min}-{max}`.
        bins_range_area: Sorted list of area bin edges.
        method_computer: Class implementing `compute_metrics`, instantiated once
            per bin (e.g. one of the `*_object_scale_local_computer` classes).
        reduction_basis: Either "pred", "ref" or None, forwarded to `method_computer`
            when not None.
        include_outer_bins: If True, adds an unbounded bin below the first edge
            and above the last edge.

    Raises:
        AssertionError: If `reduction_basis` is not "pred", "ref" or None.
    """

    def __init__(
        self,
        name_change: str,
        bins_range_area: list,
        method_computer: Any,
        reduction_basis: Optional[str],
        include_outer_bins: bool = True,
    ) -> None:
        self.name_change = name_change
        self.bins_range_area = (
            [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        )
        self.method_computer = method_computer
        self.reduction_basis = reduction_basis
        assert self.reduction_basis in [
            "pred",
            "ref",
            None,
        ], "reduction_basis must be 'pred' or 'ref' or None"

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Dict[str, Any]:
        """Computes the wrapped metric for each area bin.

        Args:
            images: Dictionary of images, must contain one entry per area bin.
            metrics_previous: Previously computed local metrics, forwarded as-is.
            row: GeoDataFrame row corresponding to the sample, forwarded as-is.

        Returns:
            Dictionary mapping each bin name (e.g. "0-100") to the metric value
            computed by `method_computer` for that bin.
        """
        dict_metrics = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]
            max_area_current_str = "" if max_area_current is None else str(int(max_area_current))
            min_area_current_str = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current_str}-{max_area_current_str}"
            name_change_bin = f"{self.name_change}_{name_bin}"

            if self.reduction_basis is None:
                method_computer = self.method_computer(name_change_bin)
            else:
                method_computer = self.method_computer(name_change_bin, self.reduction_basis)
            metric_value = method_computer.compute_metrics(images, metrics_previous, row)

            dict_metrics[name_bin] = metric_value

        return dict_metrics


class mean_local_computer:
    """Computes a mean (optionally its square root) from a (sum, count) component metric.

    Args:
        metric_component_name: Name of the previously computed `(sum, nb_value)`
            metric to reduce.
        root: If True, returns the square root of the mean (e.g. to derive RMSE
            from a mean squared error component).
    """

    def __init__(self, metric_component_name: str, root: bool = False) -> None:
        self.metric_component_name = metric_component_name
        self.root = root

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> float:
        """Computes the mean of the referenced component metric.

        Args:
            images: Dictionary of images (unused).
            metrics_previous: Previously computed local metrics, must contain
                `metric_component_name` as a `(sum, nb_value)` tuple.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            The mean value (or its square root if `root` is True), or NaN if
            `nb_value` is 0.

        Raises:
            Exception: If `metric_component_name` is not present in `metrics_previous`.
        """
        if self.metric_component_name not in metrics_previous:
            raise Exception(f"You must first load {self.metric_component_name}")

        metric_component = metrics_previous[self.metric_component_name]

        # if there are no value, the mean is nan, we don't want to take it into account, the problem is due to the target mask
        if metric_component[1] == 0:
            return np.nan

        mean = metric_component[0] / metric_component[1]
        if self.root:
            return np.sqrt(mean)
        else:
            return mean


class mae_component_computer:
    """Computes the sum of absolute errors and count of valid pixels for MAE.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
        min_value_threshold_or: Minimum value; a pixel is kept if either the
            prediction or the target is above this threshold.
        max_value_threshold_or: Maximum value; a pixel is kept if either the
            prediction or the target is below this threshold.
        min_value_threshold_and: Minimum value; a pixel is kept only if both the
            prediction and the target are above this threshold.
        max_value_threshold_and: Maximum value; a pixel is kept only if both the
            prediction and the target are below this threshold.

    Raises:
        AssertionError: If both `*_or` and `*_and` thresholds are set simultaneously.
    """

    def __init__(
        self,
        name_pred: str,
        name_target: str,
        min_value_threshold_or: Optional[float] = None,
        max_value_threshold_or: Optional[float] = None,
        min_value_threshold_and: Optional[float] = None,
        max_value_threshold_and: Optional[float] = None,
    ) -> None:
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_value_threshold_or = min_value_threshold_or
        self.max_value_threshold_or = max_value_threshold_or
        self.min_value_threshold_and = min_value_threshold_and
        self.max_value_threshold_and = max_value_threshold_and

        # We want to assert that we have either "or" thresholds, "and" thresholds, or none, but not both at the same time
        assert not (
            (self.min_value_threshold_or is not None or self.max_value_threshold_or is not None)
            and (
                self.min_value_threshold_and is not None
                or self.max_value_threshold_and is not None
            )
        ), (
            "You cannot use both *_or and *_and thresholds simultaneously. "
            "Choose either *_or parameters, *_and parameters, or none, but not both."
        )

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[float, int]:
        """Computes the sum of absolute differences and the count of valid pixels.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A `(sum_absolute_differences, nb_value)` tuple.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]

        valid_mask = ~(np.isnan(image_pred) | np.isnan(image_target))
        if self.min_value_threshold_or is not None:
            valid_mask = valid_mask & (
                (image_pred >= self.min_value_threshold_or)
                | (image_target >= self.min_value_threshold_or)
            )
        if self.max_value_threshold_or is not None:
            valid_mask = valid_mask & (
                (image_pred <= self.max_value_threshold_or)
                | (image_target <= self.max_value_threshold_or)
            )
        if self.min_value_threshold_and is not None:
            valid_mask = valid_mask & (
                (image_pred >= self.min_value_threshold_and)
                & (image_target >= self.min_value_threshold_and)
            )
        if self.max_value_threshold_and is not None:
            valid_mask = valid_mask & (
                (image_pred <= self.max_value_threshold_and)
                & (image_target <= self.max_value_threshold_and)
            )

        image_pred = image_pred[valid_mask]
        image_target = image_target[valid_mask]

        absolute_differences = np.abs(image_pred - image_target)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(valid_mask)
        return sum, nb_value


class rmse_component_computer:
    """Computes the sum of squared errors and count of valid pixels for RMSE.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
    """

    def __init__(self, name_pred: str, name_target: str) -> None:
        self.name_pred = name_pred
        self.name_target = name_target

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[float, int]:
        """Computes the sum of squared differences and the count of valid pixels.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A `(sum_squared_differences, nb_value)` tuple.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]

        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]

        squared_difference = np.square(image_pred - image_target)
        sum = np.sum(squared_difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value


class me_component_computer:
    """Computes the sum of signed errors and count of valid pixels for the mean error.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
    """

    def __init__(self, name_pred: str, name_target: str) -> None:
        self.name_pred = name_pred
        self.name_target = name_target

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[float, int]:
        """Computes the sum of signed differences (pred - target) and the count of valid pixels.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A `(sum_differences, nb_value)` tuple.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]

        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]

        difference = image_pred - image_target
        sum = np.sum(difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value


class nmae_component_computer:
    """Computes the sum of normalized absolute errors and count of valid pixels.

    The normalization divides each absolute difference by `target + 1`, and
    pixels with a target below `min_target` are excluded.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
        min_target: Minimum target value for a pixel to be considered valid.
    """

    def __init__(self, name_pred: str, name_target: str, min_target: float) -> None:
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_target = min_target

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[float, int]:
        """Computes the sum of normalized absolute differences and the count of valid pixels.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A `(sum_normalized_absolute_differences, nb_value)` tuple.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]

        unvalid_mask = (
            np.isnan(image_pred) | np.isnan(image_target) | (image_target < self.min_target)
        )
        image_pred = image_pred[~unvalid_mask]
        image_target = image_target[~unvalid_mask]

        absolute_differences = np.abs(image_pred - image_target) / (image_target + 1)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~unvalid_mask)
        return sum, nb_value


class iou_local_computer:
    """Computes the intersection and union pixel counts between two binary masks.

    Args:
        name_pred: Name of the predicted mask image in `images`.
        name_target: Name of the target mask image in `images`.
    """

    def __init__(self, name_pred: str, name_target: str) -> None:
        self.name_pred = name_pred
        self.name_target = name_target

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[int, int]:
        """Computes the intersection and union of the two masks over valid pixels.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            An `(intersection, union)` tuple of pixel counts.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target not in images):
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        mask_pred = images[self.name_pred]
        mask_target = images[self.name_target]

        nan_mask = np.isnan(mask_pred) | np.isnan(mask_target)
        mask_pred = mask_pred[~nan_mask].astype(bool)
        mask_target = mask_target[~nan_mask].astype(bool)

        intersection = np.sum(mask_pred & mask_target)
        union = np.sum(mask_pred | mask_target)

        return intersection, union


class precision_recall_component_object_scale_local_computer:
    """Computes true/false positive and negative counts from object-scale evaluation masks.

    Args:
        name_mask: Base name of the evaluation mask; reads `{name_mask}_pred`
            and `{name_mask}_ref`.
    """

    def __init__(self, name_mask: str) -> None:
        self.name_mask = name_mask

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Dict[str, int]:
        """Computes the precision/recall components from the pred and ref evaluation masks.

        Args:
            images: Dictionary of images, must contain `{name_mask}_pred` and
                `{name_mask}_ref`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Dictionary with `true_positive_pred`, `false_positive`,
            `true_positive_ref` and `false_negative` pixel counts.

        Raises:
            Exception: If the required masks are not present in `images`.
        """
        name_pred = f"{self.name_mask}_pred"
        name_ref = f"{self.name_mask}_ref"
        if (name_pred not in images) or (name_ref not in images):
            raise Exception(f"You must first load {name_pred} and {name_ref}")

        mask_pred = images[name_pred]
        mask_ref = images[name_ref]

        # In the evaluation masks, 1 is a true positive and -1 is a false positive (pred side) / false negative (ref side)
        true_positive_pred = np.sum(mask_pred == 1)
        false_positive = np.sum(mask_pred == -1)
        true_positive_ref = np.sum(mask_ref == 1)
        false_negative = np.sum(mask_ref == -1)

        return {
            "true_positive_pred": true_positive_pred,
            "false_positive": false_positive,
            "true_positive_ref": true_positive_ref,
            "false_negative": false_negative,
        }


class flatten_local_computer:
    """Extracts flattened, thresholded prediction and target arrays for later aggregation.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
        min_value_threshold_or: Minimum value; a pixel is kept if either the
            prediction or the target is above this threshold.
        max_value_threshold_or: Maximum value; a pixel is kept if either the
            prediction or the target is below this threshold.
        min_value_threshold_and: Minimum value; a pixel is kept only if both the
            prediction and the target are above this threshold.
        max_value_threshold_and: Maximum value; a pixel is kept only if both the
            prediction and the target are below this threshold.

    Raises:
        AssertionError: If both `*_or` and `*_and` thresholds are set simultaneously.
    """

    def __init__(
        self,
        name_pred: str,
        name_target: str,
        min_value_threshold_or: Optional[float] = None,
        max_value_threshold_or: Optional[float] = None,
        min_value_threshold_and: Optional[float] = None,
        max_value_threshold_and: Optional[float] = None,
    ) -> None:
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_value_threshold_or = min_value_threshold_or
        self.max_value_threshold_or = max_value_threshold_or
        self.min_value_threshold_and = min_value_threshold_and
        self.max_value_threshold_and = max_value_threshold_and

        # We want to assert that we have either "or" thresholds, "and" thresholds, or none, but not both at the same time
        assert not (
            (self.min_value_threshold_or is not None or self.max_value_threshold_or is not None)
            and (
                self.min_value_threshold_and is not None
                or self.max_value_threshold_and is not None
            )
        ), (
            "You cannot use both *_or and *_and thresholds simultaneously. "
            "Choose either *_or parameters, *_and parameters, or none, but not both."
        )

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters valid pixels according to the configured thresholds and flattens them.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A `(flattened_target, flattened_pred)` tuple of 1D arrays.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if self.name_pred not in images or self.name_target not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]

        valid_mask = ~(np.isnan(image_pred) | np.isnan(image_target))
        if self.min_value_threshold_or is not None:
            valid_mask = valid_mask & (
                (image_pred >= self.min_value_threshold_or)
                | (image_target >= self.min_value_threshold_or)
            )
        if self.max_value_threshold_or is not None:
            valid_mask = valid_mask & (
                (image_pred <= self.max_value_threshold_or)
                | (image_target <= self.max_value_threshold_or)
            )

        if self.min_value_threshold_and is not None:
            valid_mask = valid_mask & (
                (image_pred >= self.min_value_threshold_and)
                & (image_target >= self.min_value_threshold_and)
            )
        if self.max_value_threshold_and is not None:
            valid_mask = valid_mask & (
                (image_pred <= self.max_value_threshold_and)
                & (image_target <= self.max_value_threshold_and)
            )

        image_pred = image_pred[valid_mask]
        image_target = image_target[valid_mask]

        return image_target.flatten(), image_pred.flatten()


class group_by_bins_local_computer:
    """Groups per-pixel metric values by target-value bins.

    Args:
        name_pred: Name of the prediction image in `images`.
        name_target: Name of the target image in `images`.
        bins: Sorted list of bin edges applied to the target values.
        method_metrics: Name of one of this class's methods (e.g.
            `"absolute_error"`) used to compute the per-pixel metric within
            each bin.
    """

    def __init__(self, name_pred: str, name_target: str, bins: list, method_metrics: str) -> None:
        self.name_pred = name_pred
        self.name_target = name_target
        self.bins = bins
        self.method_metrics = getattr(self, method_metrics)

    def absolute_error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise absolute error between prediction and target."""
        return np.abs(pred - target)

    def squared_error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise squared error between prediction and target."""
        return np.square(pred - target)

    def error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise signed error (pred - target)."""
        return pred - target

    def keep_pred_positive(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the prediction values greater than -5 (target is unused)."""
        mask = pred > -5
        return pred[mask]

    def return_pred(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the prediction values unchanged (target is unused)."""
        return pred

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> list:
        """Computes the configured per-pixel metric within each target-value bin.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A list with one flattened array per bin (empty array if the bin has
            no valid pixels).

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        target = images[self.name_target]
        pred = images[self.name_pred]

        nan_mask = np.isnan(target) | np.isnan(pred)

        target = target[~nan_mask]
        pred = pred[~nan_mask]

        bin_metrics = []
        for i in range(len(self.bins) - 1):
            mask_bin = (target >= self.bins[i]) & (target < self.bins[i + 1])
            if mask_bin.sum() > 0:
                bin_metric = self.method_metrics(pred[mask_bin], target[mask_bin]).flatten()
            else:
                bin_metric = np.array([])
            bin_metrics.append(bin_metric)

        return bin_metrics


class group_by_nb_image_local_computer:
    """Computer that groups metrics by bins of number of images used per sample.

    This class processes images to compute metrics for samples grouped by the
    number of images (e.g. Sentinel acquisitions) that were used to build them,
    allowing for analysis of predictions versus targets as a function of image count.

    Args:
        name_pred: Name of the prediction image in the images dictionary.
        name_target: Name of the target image in the images dictionary.
        bins_nb_image: List of `(min, max)` tuples defining the number-of-image ranges.
        vrts_column: Name of the row column containing the list of source images
            (its length gives the number of images).
        method_metrics: Name of one of this class's methods (e.g.
            `"absolute_error"`) used to compute metrics between predictions and targets.
    """

    def __init__(
        self,
        name_pred: str,
        name_target: str,
        bins_nb_image: list,
        vrts_column: str,
        method_metrics: str,
    ) -> None:
        self.name_pred = name_pred
        self.name_target = name_target
        self.bins_nb_image = bins_nb_image
        self.vrts_column = vrts_column
        self.method_metrics = getattr(self, method_metrics)

    def absolute_error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise absolute error between prediction and target."""
        return np.abs(pred - target)

    def squared_error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise squared error between prediction and target."""
        return np.square(pred - target)

    def error(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the elementwise signed error (pred - target)."""
        return pred - target

    def compute_metrics(
        self, images: Dict[str, np.ndarray], metrics_previous: Dict[str, Any], row: pd.Series
    ) -> list:
        """Computes the configured metric for the bin matching the sample's number of images.

        Args:
            images: Dictionary of images, must contain `name_pred` and `name_target`.
            metrics_previous: Previously computed local metrics (unused).
            row: GeoDataFrame row corresponding to the sample; used to read
                `vrts_column` and derive the number of images.

        Returns:
            A list with one entry per bin in `bins_nb_image`: the computed
            metric for the bin containing the sample's image count, and an
            empty array for all other bins.

        Raises:
            Exception: If `name_pred` or `name_target` is not present in `images`.
        """
        if (self.name_pred not in images) or (self.name_target) not in images:
            raise Exception(f"You must first load {self.name_pred} and {self.name_target}")

        target = images[self.name_target]
        pred = images[self.name_pred]

        nan_mask = np.isnan(target) | np.isnan(pred)
        target = target[~nan_mask]
        pred = pred[~nan_mask]

        nb_image = len(row[self.vrts_column])

        bin_metrics = []
        for bin in self.bins_nb_image:
            if (nb_image >= int(bin[0])) and (
                nb_image <= int(bin[1])
            ):  # check if the month is in the bin
                bin_metrics.append(self.method_metrics(pred, target))
            else:
                bin_metrics.append(np.array([]))

        return bin_metrics
