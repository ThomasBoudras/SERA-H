import numpy as np


class get_metrics_global:
    def __init__(self, metrics_set, metrics_to_save, output_path_xlsx) :
        self.metrics_set = metrics_set
        self.metrics_to_save = metrics_to_save
        self.output_path_xlsx = output_path_xlsx

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

class precision_global_computer:
    def __init__(self, name_true_positive, name_false_positive):
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive not in metrics_local) or (self.name_false_positive not in metrics_local) :
            raise Exception(f"You must first compute {self.name_true_positive} and {self.name_false_positive}")

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_positive = np.sum(metrics_local[self.name_false_positive])

        # case where there are no predictions, the precision is impossible to compute, so we return nan
        if true_positive + false_positive == 0:
            return np.nan
        
        precision = true_positive / (true_positive + false_positive)
        return precision

 
class  recall_global_computer :
    def __init__(self, name_true_positive, name_false_negative):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive not in metrics_local) or (self.name_false_negative not in metrics_local) :
            raise Exception(f"You must first compute {self.name_true_positive} and {self.name_false_negative}")

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_negative = np.sum(metrics_local[self.name_false_negative])

        # case where there are no targets, the recall is impossible to compute, so we return nan
        if true_positive + false_negative == 0:
            return np.nan
        
        recall = true_positive / (true_positive + false_negative)
        return recall
    

class f1_score_global_computer:
    def __init__(self, name_true_positive_precision, name_true_positive_recall, name_false_negative, name_false_positive):
        self.name_true_positive_precision = name_true_positive_precision
        self.name_true_positive_recall = name_true_positive_recall

        self.name_false_negative = name_false_negative
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive_precision not in metrics_local) or (self.name_true_positive_recall not in metrics_local) or (self.name_false_negative not in metrics_local) or (self.name_false_positive not in metrics_local) :
            raise Exception(f"You must first compute {self.name_true_positive_precision}, {self.name_true_positive_recall}, {self.name_false_negative} and {self.name_false_positive}")

        true_positive_precision = np.sum(metrics_local[self.name_true_positive_precision])
        true_positive_recall = np.sum(metrics_local[self.name_true_positive_recall])
        false_negative = np.sum(metrics_local[self.name_false_negative])
        false_positive = np.sum(metrics_local[self.name_false_positive])
                
        # if there are no positive predictions or no positive targets, the f1 score is 1.0, the model correctly predicts no changes
        if true_positive_precision + true_positive_recall + false_negative + false_positive == 0 :
            return 1.0

        # if one of the true positive metrics is 0, none of the prediction was correct or none of the target was predicted, so the f1 score is 0 (normally if one of the true positive metrics is 0, the other is also 0)
        if true_positive_precision == 0 or true_positive_recall == 0 :
            return 0.0
        
        recall = true_positive_recall / (true_positive_recall + false_negative)
        precision = true_positive_precision / (true_positive_precision + false_positive)
        
        f1_score = (2 * recall * precision) / (recall + precision)
        return f1_score


class binned_by_area_precision_global_computer:
    def __init__(self, name_true_positive, name_false_positive, bins_range_area, include_outer_bins=True):
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive
        self.bins_range_area = [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        
    def compute_metrics(self, metrics_local) :
        dict_precision = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]
            max_area_current = "" if max_area_current is None else str(int(max_area_current))
            min_area_current = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current}-{max_area_current}"

            name_true_positive_bin = f"{self.name_true_positive}_{name_bin}"
            name_false_positive_bin = f"{self.name_false_positive}_{name_bin}"

            precision_computer = precision_global_computer(name_true_positive_bin, name_false_positive_bin)
            precision = precision_computer.compute_metrics(metrics_local)
            dict_precision[name_bin] = precision
            
        return dict_precision 


class binned_by_area_recall_global_computer:
    def __init__(self, name_true_positive, name_false_negative, bins_range_area, include_outer_bins=True):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        self.bins_range_area = [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        
    def compute_metrics(self, metrics_local) :
        dict_recall = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]
            max_area_current = "" if max_area_current is None else str(int(max_area_current))
            min_area_current = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current}-{max_area_current}"

            name_true_positive_bin = f"{self.name_true_positive}_{name_bin}"
            name_false_negative_bin = f"{self.name_false_negative}_{name_bin}"

            recall_computer = recall_global_computer(name_true_positive_bin, name_false_negative_bin)
            recall = recall_computer.compute_metrics(metrics_local)
            dict_recall[name_bin] = recall
            
        return dict_recall 



class mean_global_computeur :
    def __init__(self, name_metric, root=False):
        self.name_metric =name_metric 
        self.root = root
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            raise Exception(f"You must first compute {self.name_metric}")
        
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
            raise Exception(f"You must first compute {self.name_metric}")
        
        return np.sum([value[1] for value in metrics_local[self.name_metric]])

class concat_tuples_global_computer:
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            raise Exception(f"You must first compute {self.name_metric}")
        
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
            raise Exception(f"You must first compute {self.name_metric}")
        
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
            raise Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]
        first_element = metric[0]
        if isinstance(first_element, (list, tuple, np.ndarray)):
            return tuple([np.std(np.concatenate([value[i] for value in metric])) for i in range(len(first_element))])
        else:
            return np.std(metric)