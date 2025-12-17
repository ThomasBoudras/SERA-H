import numpy as np


class get_metrics_global:
    def __init__(self, metrics_set, metrics_to_print) :
        self.metrics_set = metrics_set
        self.metrics_to_print = metrics_to_print

    def __call__(self, metrics_local):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metrics[metric_name] =  metric_computer.compute_metrics(metrics_local)
        return metrics


class mean_global_computeur :
    def __init__(self, name_metric, root=False):
        self.name_metric =name_metric 
        self.root = root
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]

        sum = np.sum([value[0] for value in metric])
        nb_value = np.sum([value[1] for value in metric])

        if self.root :
            return np.sqrt(sum/nb_value)
        return sum/nb_value
    
class nb_values_global_computer :
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        return np.sum([value[1] for value in metrics_local[self.name_metric]])
