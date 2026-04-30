import numpy as np
import json
from skimage.measure import label

class get_metrics_local:
    def __init__(self, metrics_set) :
        self.metrics_set = metrics_set

    def __call__(self, images, row):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metric_value =  metric_computer.compute_metrics(images, metrics, row)
            if isinstance(metric_value, dict):
                for key, value in metric_value.items():
                    metrics[f"{metric_name}_{key}"] = value
            else:
                metrics[metric_name] = metric_value
        return metrics


class mean_local_computer:
    def __init__(self, metric_component_name, root=False):
        self.metric_component_name = metric_component_name
        self.root = root
        
    def compute_metrics(self, images, metrics_previous, row) :
        if self.metric_component_name not in metrics_previous :
            Exception(f"You must first load {self.metric_component_name}")
        
        metric_component = metrics_previous[self.metric_component_name]

        # if there are no value, the mean is nan, we don't want to take it into account, the problem is due to the target mask
        if metric_component[1] == 0 :
            return np.nan

        mean = metric_component[0]/metric_component[1]
        if self.root :
            return np.sqrt(mean)
        else :
            return mean
        
        
class mae_component_computer:
    def __init__(self, name_pred, name_target, min_value_threshold_or=None, max_value_threshold_or=None, min_value_threshold_and=None, max_value_threshold_and=None):
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_value_threshold_or = min_value_threshold_or
        self.max_value_threshold_or = max_value_threshold_or
        self.min_value_threshold_and = min_value_threshold_and
        self.max_value_threshold_and = max_value_threshold_and

        # We want to assert that we have either "or" thresholds, "and" thresholds, or none, but not both at the same time
        assert not (
            (self.min_value_threshold_or is not None or self.max_value_threshold_or is not None)
            and
            (self.min_value_threshold_and is not None or self.max_value_threshold_and is not None)
        ), (
            "You cannot use both *_or and *_and thresholds simultaneously. "
            "Choose either *_or parameters, *_and parameters, or none, but not both."
        )

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        valid_mask = ~(np.isnan(image_pred) | np.isnan(image_target))
        if self.min_value_threshold_or is not None :
            valid_mask = valid_mask & ((image_pred >= self.min_value_threshold_or) | (image_target >= self.min_value_threshold_or))
        if self.max_value_threshold_or is not None :
            valid_mask = valid_mask & ((image_pred <= self.max_value_threshold_or) | (image_target <= self.max_value_threshold_or))
        if self.min_value_threshold_and is not None :
            valid_mask = valid_mask & ((image_pred >= self.min_value_threshold_and) & (image_target >= self.min_value_threshold_and))
        if self.max_value_threshold_and is not None :
            valid_mask = valid_mask & ((image_pred <= self.max_value_threshold_and) & (image_target <= self.max_value_threshold_and))

        image_pred = image_pred[valid_mask]
        image_target = image_target[valid_mask]
        
        absolute_differences = np.abs(image_pred - image_target)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(valid_mask)
        return sum, nb_value


class rmse_component_computer:
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

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
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
           
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

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
    def __init__(self, name_pred, name_target, min_target):
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_target = min_target

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
            
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        unvalid_mask = np.isnan(image_pred) | np.isnan(image_target) | (image_target < self.min_target)
        image_pred = image_pred[~unvalid_mask]
        image_target = image_target[~unvalid_mask]
        
        absolute_differences = np.abs(image_pred - image_target)/(image_target + 1)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~unvalid_mask)
        return sum, nb_value


class treecover_local_computer:
    def __init__(self, name_pred, name_target, threshold):
        self.name_pred = name_pred
        self.name_target = name_target
        self.threshold = threshold

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]

        mask_pred = image_pred > self.threshold
        mask_target = image_target > self.threshold
        
        intersection = np.sum(mask_pred & mask_target)
        union = np.sum(mask_pred | mask_target)
        
        return intersection, union


class flatten_local_computer:
    def __init__(self, name_pred, name_target, min_value_threshold_or=None, max_value_threshold_or=None, min_value_threshold_and=None, max_value_threshold_and=None):
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_value_threshold_or = min_value_threshold_or
        self.max_value_threshold_or = max_value_threshold_or
        self.min_value_threshold_and = min_value_threshold_and
        self.max_value_threshold_and = max_value_threshold_and

        # We want to assert that we have either "or" thresholds, "and" thresholds, or none, but not both at the same time
        assert not (
            (self.min_value_threshold_or is not None or self.max_value_threshold_or is not None)
            and
            (self.min_value_threshold_and is not None or self.max_value_threshold_and is not None)
        ), (
            "You cannot use both *_or and *_and thresholds simultaneously. "
            "Choose either *_or parameters, *_and parameters, or none, but not both."
        )

    def compute_metrics(self, images, metrics_previous, row) :
        if self.name_pred not in images or self.name_target not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        valid_mask = ~(np.isnan(image_pred) | np.isnan(image_target))
        if self.min_value_threshold_or is not None :
            valid_mask = valid_mask & ((image_pred >= self.min_value_threshold_or) | (image_target >= self.min_value_threshold_or))
        if self.max_value_threshold_or is not None :
            valid_mask = valid_mask & ((image_pred <= self.max_value_threshold_or) | (image_target <= self.max_value_threshold_or))
        
        if self.min_value_threshold_and is not None :
            valid_mask = valid_mask & ((image_pred >= self.min_value_threshold_and) & (image_target >= self.min_value_threshold_and))
        if self.max_value_threshold_and is not None :
            valid_mask = valid_mask & ((image_pred <= self.max_value_threshold_and) & (image_target <= self.max_value_threshold_and))


        image_pred = image_pred[valid_mask]
        image_target = image_target[valid_mask]
        
        return image_target.flatten(), image_pred.flatten()

    
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        target = images[self.name_target]
        pred = images[self.name_pred]

        nan_mask = np.isnan(target) | np.isnan(pred)

        target = target[~nan_mask]
        pred = pred[~nan_mask]


        bin_metrics = []
        for i in range(len(self.bins)-1):
            mask_bin = (target >= self.bins[i]) & (target < self.bins[i+1])
            if mask_bin.sum() > 0:
                bin_metric = self.method_metrics(pred[mask_bin], target[mask_bin]).flatten()
            else :
                bin_metric = np.array([])
            bin_metrics.append(bin_metric)

        return bin_metrics


class group_by_nb_image_local_computer:
    """
    Computer that groups metrics by date bins based on a specified date column.
    
    This class processes images to compute metrics for specific date ranges,
    allowing for temporal analysis of predictions versus targets.
    
    Args:
        name_pred (str): Name of the prediction image in the images dictionary
        name_target (str): Name of the target image in the images dictionary
        name_base (str): Base name for the output metrics
        bins_nb_image (list): List of date bin tuples defining the date ranges
        vrts_column (str): Name of the column containing date information
        method_metrics (callable): Method to compute metrics between predictions and targets
    """
    def __init__(self, name_pred, name_target, bins_nb_image, vrts_column, method_metrics):
        self.name_pred = name_pred
        self.name_target = name_target
        self.bins_nb_image = bins_nb_image
        self.vrts_column = vrts_column 
        self.method_metrics = getattr(self, method_metrics)

    def absolute_error(self, pred, target):
        return np.abs(pred - target)
    
    def squared_error(self, pred, target):
        return np.square(pred - target)
    
    def error(self, pred, target):
        return pred - target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        target = images[self.name_target]
        pred = images[self.name_pred]
        
        nan_mask = np.isnan(target) | np.isnan(pred)
        target = target[~nan_mask]
        pred = pred[~nan_mask]
        
        nb_image = len(row[self.vrts_column])

        bin_metrics = []
        for bin in self.bins_nb_image:
            if (nb_image >= int(bin[0])) and (nb_image <= int(bin[1])):  # check if the month is in the bin
                bin_metrics.append(self.method_metrics(pred, target))
            else :
                bin_metrics.append(np.array([]))
        
        return bin_metrics
  