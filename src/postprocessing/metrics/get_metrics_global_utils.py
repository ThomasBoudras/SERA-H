"""Computers that aggregate local (per-sample) metrics into global metrics.

Each computer implements a `compute_metrics` method taking the dictionary of
aggregated local metrics and returns either a scalar value or a dict of named
values, which `get_metrics_global` combines into a single metrics dictionary.
"""

from typing import Any, Dict, Tuple, Union

import numpy as np


class get_metrics_global:
    """Runs a set of global metric computers on the aggregated local metrics.

    Args:
        metrics_set: Mapping from metric name to a computer object exposing a
            `compute_metrics(metrics_local)` method.
        metrics_to_save: Either `"all"`, a list of metric names to keep, or
            `None`/falsy to save nothing.
        output_path_xlsx: Path to the Excel file where selected metrics are saved.
    """

    def __init__(
        self,
        metrics_set: Dict[str, Any],
        metrics_to_save: Union[str, list, None],
        output_path_xlsx: str,
    ) -> None:
        self.metrics_set = metrics_set
        self.metrics_to_save = metrics_to_save
        self.output_path_xlsx = output_path_xlsx

    def __call__(self, metrics_local: Dict[str, list]) -> Dict[str, Any]:
        """Computes all configured global metrics from the aggregated local metrics.

        Args:
            metrics_local: Dictionary mapping local metric names to the list of
                per-sample values aggregated across the dataset.

        Returns:
            Dictionary mapping metric names to their computed values. When a
            computer returns a dict, its keys are prefixed with the metric name.
        """
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metric_value = metric_computer.compute_metrics(metrics_local)
            if isinstance(metric_value, dict):
                for key, value in metric_value.items():
                    metrics[f"{metric_name}_{key}"] = value
            else:
                metrics[metric_name] = metric_value
        return metrics


class precision_global_computer:
    """Computes the global precision from aggregated true/false positive counts.

    Args:
        name_true_positive: Name of the local metric holding per-sample
            true-positive counts.
        name_false_positive: Name of the local metric holding per-sample
            false-positive counts.
    """

    def __init__(self, name_true_positive: str, name_false_positive: str) -> None:
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive

    def compute_metrics(self, metrics_local: Dict[str, list]) -> float:
        """Computes precision as true_positive / (true_positive + false_positive).

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_true_positive` and `name_false_positive`.

        Returns:
            The precision value, or NaN if there are no predictions.

        Raises:
            Exception: If the required metrics are not present in `metrics_local`.
        """
        if (self.name_true_positive not in metrics_local) or (
            self.name_false_positive not in metrics_local
        ):
            raise Exception(
                f"You must first compute {self.name_true_positive} and {self.name_false_positive}"
            )

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_positive = np.sum(metrics_local[self.name_false_positive])

        # case where there are no predictions, the precision is impossible to compute, so we return nan
        if true_positive + false_positive == 0:
            return np.nan

        precision = true_positive / (true_positive + false_positive)
        return precision


class recall_global_computer:
    """Computes the global recall from aggregated true positive/false negative counts.

    Args:
        name_true_positive: Name of the local metric holding per-sample
            true-positive counts.
        name_false_negative: Name of the local metric holding per-sample
            false-negative counts.
    """

    def __init__(self, name_true_positive: str, name_false_negative: str) -> None:
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative

    def compute_metrics(self, metrics_local: Dict[str, list]) -> float:
        """Computes recall as true_positive / (true_positive + false_negative).

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_true_positive` and `name_false_negative`.

        Returns:
            The recall value, or NaN if there are no targets.

        Raises:
            Exception: If the required metrics are not present in `metrics_local`.
        """
        if (self.name_true_positive not in metrics_local) or (
            self.name_false_negative not in metrics_local
        ):
            raise Exception(
                f"You must first compute {self.name_true_positive} and {self.name_false_negative}"
            )

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_negative = np.sum(metrics_local[self.name_false_negative])

        # case where there are no targets, the recall is impossible to compute, so we return nan
        if true_positive + false_negative == 0:
            return np.nan

        recall = true_positive / (true_positive + false_negative)
        return recall


class f1_score_global_computer:
    """Computes the global F1 score from separately-tracked precision/recall true positives.

    Precision and recall true positives can differ because true positives are
    computed independently on each side (e.g. object-scale matching).

    Args:
        name_true_positive_precision: Name of the local metric holding the
            true-positive counts used for precision.
        name_true_positive_recall: Name of the local metric holding the
            true-positive counts used for recall.
        name_false_negative: Name of the local metric holding false-negative counts.
        name_false_positive: Name of the local metric holding false-positive counts.
    """

    def __init__(
        self,
        name_true_positive_precision: str,
        name_true_positive_recall: str,
        name_false_negative: str,
        name_false_positive: str,
    ) -> None:
        self.name_true_positive_precision = name_true_positive_precision
        self.name_true_positive_recall = name_true_positive_recall

        self.name_false_negative = name_false_negative
        self.name_false_positive = name_false_positive

    def compute_metrics(self, metrics_local: Dict[str, list]) -> float:
        """Computes the F1 score from the aggregated precision/recall components.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_true_positive_precision`, `name_true_positive_recall`,
                `name_false_negative` and `name_false_positive`.

        Returns:
            The F1 score. Returns 1.0 when there are no positive predictions
            and no positive targets, and 0.0 when either true-positive count
            is 0 while the other counts are non-zero.

        Raises:
            Exception: If any of the required metrics is not present in `metrics_local`.
        """
        if (
            (self.name_true_positive_precision not in metrics_local)
            or (self.name_true_positive_recall not in metrics_local)
            or (self.name_false_negative not in metrics_local)
            or (self.name_false_positive not in metrics_local)
        ):
            raise Exception(
                f"You must first compute {self.name_true_positive_precision}, {self.name_true_positive_recall}, {self.name_false_negative} and {self.name_false_positive}"
            )

        true_positive_precision = np.sum(metrics_local[self.name_true_positive_precision])
        true_positive_recall = np.sum(metrics_local[self.name_true_positive_recall])
        false_negative = np.sum(metrics_local[self.name_false_negative])
        false_positive = np.sum(metrics_local[self.name_false_positive])

        # if there are no positive predictions or no positive targets, the f1 score is 1.0, the model correctly predicts no changes
        if true_positive_precision + true_positive_recall + false_negative + false_positive == 0:
            return 1.0

        # if one of the true positive metrics is 0, none of the prediction was correct or none of the target was predicted, so the f1 score is 0 (normally if one of the true positive metrics is 0, the other is also 0)
        if true_positive_precision == 0 or true_positive_recall == 0:
            return 0.0

        recall = true_positive_recall / (true_positive_recall + false_negative)
        precision = true_positive_precision / (true_positive_precision + false_positive)

        f1_score = (2 * recall * precision) / (recall + precision)
        return f1_score


class binned_by_area_precision_global_computer:
    """Computes global precision independently for a series of area bins.

    Args:
        name_true_positive: Base name of the local metric holding per-bin
            true-positive counts (`{name_true_positive}_{bin}`).
        name_false_positive: Base name of the local metric holding per-bin
            false-positive counts (`{name_false_positive}_{bin}`).
        bins_range_area: Sorted list of area bin edges.
        include_outer_bins: If True, adds an unbounded bin below the first edge
            and above the last edge.
    """

    def __init__(
        self,
        name_true_positive: str,
        name_false_positive: str,
        bins_range_area: list,
        include_outer_bins: bool = True,
    ) -> None:
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive
        self.bins_range_area = (
            [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        )

    def compute_metrics(self, metrics_local: Dict[str, list]) -> Dict[str, float]:
        """Computes the precision for each area bin.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                one true-positive and false-positive entry per bin.

        Returns:
            Dictionary mapping each bin name (e.g. "0-100") to its precision value.
        """
        dict_precision = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]
            max_area_current = "" if max_area_current is None else str(int(max_area_current))
            min_area_current = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current}-{max_area_current}"

            name_true_positive_bin = f"{self.name_true_positive}_{name_bin}"
            name_false_positive_bin = f"{self.name_false_positive}_{name_bin}"

            precision_computer = precision_global_computer(
                name_true_positive_bin, name_false_positive_bin
            )
            precision = precision_computer.compute_metrics(metrics_local)
            dict_precision[name_bin] = precision

        return dict_precision


class binned_by_area_recall_global_computer:
    """Computes global recall independently for a series of area bins.

    Args:
        name_true_positive: Base name of the local metric holding per-bin
            true-positive counts (`{name_true_positive}_{bin}`).
        name_false_negative: Base name of the local metric holding per-bin
            false-negative counts (`{name_false_negative}_{bin}`).
        bins_range_area: Sorted list of area bin edges.
        include_outer_bins: If True, adds an unbounded bin below the first edge
            and above the last edge.
    """

    def __init__(
        self,
        name_true_positive: str,
        name_false_negative: str,
        bins_range_area: list,
        include_outer_bins: bool = True,
    ) -> None:
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        self.bins_range_area = (
            [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        )

    def compute_metrics(self, metrics_local: Dict[str, list]) -> Dict[str, float]:
        """Computes the recall for each area bin.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                one true-positive and false-negative entry per bin.

        Returns:
            Dictionary mapping each bin name (e.g. "0-100") to its recall value.
        """
        dict_recall = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]
            max_area_current = "" if max_area_current is None else str(int(max_area_current))
            min_area_current = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current}-{max_area_current}"

            name_true_positive_bin = f"{self.name_true_positive}_{name_bin}"
            name_false_negative_bin = f"{self.name_false_negative}_{name_bin}"

            recall_computer = recall_global_computer(
                name_true_positive_bin, name_false_negative_bin
            )
            recall = recall_computer.compute_metrics(metrics_local)
            dict_recall[name_bin] = recall

        return dict_recall


class mean_global_computeur:
    """Aggregates a list of per-sample (sum, count) tuples into a global mean.

    Args:
        name_metric: Name of the local metric holding a list of `(sum, nb_value)`
            tuples, one per sample.
        root: If True, returns the square root of the mean (e.g. to derive RMSE
            from a mean squared error component).
    """

    def __init__(self, name_metric: str, root: bool = False) -> None:
        self.name_metric = name_metric
        self.root = root

    def compute_metrics(self, metrics_local: Dict[str, list]) -> float:
        """Computes the global mean from the aggregated (sum, count) tuples.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_metric` as a list of `(sum, nb_value)` tuples.

        Returns:
            The mean value (or its square root if `root` is True).

        Raises:
            Exception: If `name_metric` is not present in `metrics_local`.
        """
        if self.name_metric not in metrics_local:
            raise Exception(f"You must first compute {self.name_metric}")

        metric = metrics_local[self.name_metric]

        sum = np.sum([value[0] for value in metric])
        nb_value = np.sum([value[1] for value in metric])

        if self.root:
            return np.sqrt(sum / nb_value)
        return sum / nb_value


class nb_values_global_computer:
    """Sums the per-sample counts of a (sum, count) local metric.

    Args:
        name_metric: Name of the local metric holding a list of `(sum, nb_value)`
            tuples, one per sample.
    """

    def __init__(self, name_metric: str) -> None:
        self.name_metric = name_metric

    def compute_metrics(self, metrics_local: Dict[str, list]) -> int:
        """Sums the count component across all samples.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_metric` as a list of `(sum, nb_value)` tuples.

        Returns:
            The total number of values across all samples.

        Raises:
            Exception: If `name_metric` is not present in `metrics_local`.
        """
        if self.name_metric not in metrics_local:
            raise Exception(f"You must first compute {self.name_metric}")

        return np.sum([value[1] for value in metrics_local[self.name_metric]])


class concat_tuples_global_computer:
    """Concatenates per-sample arrays (or tuples of arrays) into global array(s).

    Args:
        name_metric: Name of the local metric holding a list of per-sample
            arrays, or a list of tuples/lists of arrays.
    """

    def __init__(self, name_metric: str) -> None:
        self.name_metric = name_metric

    def compute_metrics(
        self, metrics_local: Dict[str, list]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Concatenates the per-sample values across the whole dataset.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_metric`.

        Returns:
            A single concatenated array if the per-sample values are simple
            arrays, or a tuple of concatenated arrays (one per position) if the
            per-sample values are tuples/lists of arrays.

        Raises:
            Exception: If `name_metric` is not present in `metrics_local`.
        """
        if self.name_metric not in metrics_local:
            raise Exception(f"You must first compute {self.name_metric}")

        metric = metrics_local[self.name_metric]

        # Check if we have a list of simple values or a list of tuples/lists
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            # Case: list of tuples/lists - concatenate each position
            return tuple(
                [np.concatenate([value[i] for value in metric]) for i in range(len(first_element))]
            )
        else:
            # Case: simple list - return as single tuple element
            return np.concatenate(metric)


class mean_tuples_global_computer:
    """Computes the global mean of per-sample arrays (or tuples of arrays).

    Args:
        name_metric: Name of the local metric holding a list of per-sample
            arrays, or a list of tuples/lists of arrays.
    """

    def __init__(self, name_metric: str) -> None:
        self.name_metric = name_metric

    def compute_metrics(self, metrics_local: Dict[str, list]) -> Union[float, Tuple[float, ...]]:
        """Computes the mean over all concatenated per-sample values.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_metric`.

        Returns:
            A single mean value if the per-sample values are simple arrays, or
            a tuple of mean values (one per position) if the per-sample values
            are tuples/lists of arrays.

        Raises:
            Exception: If `name_metric` is not present in `metrics_local`.
        """
        if self.name_metric not in metrics_local:
            raise Exception(f"You must first compute {self.name_metric}")

        metric = metrics_local[self.name_metric]
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            return tuple(
                [
                    np.mean(np.concatenate([value[i] for value in metric]))
                    for i in range(len(first_element))
                ]
            )
        else:
            metric = np.concatenate(metric)
            return np.mean(metric)


class std_tuples_global_computer:
    """Computes the global standard deviation of per-sample arrays (or tuples of arrays).

    Args:
        name_metric: Name of the local metric holding a list of per-sample
            arrays, or a list of tuples/lists of arrays.
    """

    def __init__(self, name_metric: str) -> None:
        self.name_metric = name_metric

    def compute_metrics(self, metrics_local: Dict[str, list]) -> Union[float, Tuple[float, ...]]:
        """Computes the standard deviation over all concatenated per-sample values.

        Args:
            metrics_local: Dictionary of aggregated local metrics, must contain
                `name_metric`.

        Returns:
            A single standard deviation value if the per-sample values are
            simple arrays, or a tuple of standard deviation values (one per
            position) if the per-sample values are tuples/lists of arrays.

        Raises:
            Exception: If `name_metric` is not present in `metrics_local`.
        """
        if self.name_metric not in metrics_local:
            raise Exception(f"You must first compute {self.name_metric}")

        metric = metrics_local[self.name_metric]
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            return tuple(
                [
                    np.std(np.concatenate([value[i] for value in metric]))
                    for i in range(len(first_element))
                ]
            )
        else:
            return np.std(metric)
