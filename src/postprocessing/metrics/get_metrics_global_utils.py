import numpy as np


class get_metrics_global:
    def __init__(self, metrics_set, metrics_to_print) :
        self.metrics_set = metrics_set
        self.metrics_to_print = metrics_to_print

    def __call__(self, metrics_local):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metric_value =  metric_computer.compute_metrics(metrics_local)
            if isinstance(metric_value, dict):
                for key, value in metric_value.items():
                    metrics[f"{metric_name}_{key}"] = value
            else:
                metrics[metric_name] = metric_value
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

class concat_tuples_global_computer:
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]

        
        # Check if we have a list of simple values or a list of tuples/lists
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            # Case: list of tuples/lists - concatenate each position
            return tuple([np.concatenate([value[i] for value in metric]) for i in range(len(first_element))])
        else:
            # Case: simple list - return as single tuple element
            return np.concatenate(metric)

class mean_tuples_global_computer:
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            return tuple([np.mean(np.concatenate([value[i] for value in metric])) for i in range(len(first_element))])
        else:
            metric = np.concatenate(metric)
            return np.mean(metric)

class std_tuples_global_computer:
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            return tuple([np.std(np.concatenate([value[i] for value in metric])) for i in range(len(first_element))])
        else:
            return np.std(metric)