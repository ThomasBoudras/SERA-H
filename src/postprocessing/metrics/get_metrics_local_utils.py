import numpy as np
import json

class get_metrics_local:
    def __init__(self, metrics_set) :
        self.metrics_set = metrics_set

    def __call__(self, images, row):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metrics[metric_name] =  metric_computer.compute_metrics(images, metrics, row)
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
