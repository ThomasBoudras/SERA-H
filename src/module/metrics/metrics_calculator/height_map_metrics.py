import torch

class heightMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, tree_cover_threshold, min_height_nMAE):
        self.tree_cover_threshold =  tree_cover_threshold
        self.min_height_nMAE = min_height_nMAE
    
    def get_required_states(self):
        # On définit le "contrat" : voici les états dont j'ai besoin.
        # Pour chacun, je spécifie comment il doit être réduit en DDP.
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
    
    def batch_update(self, pred, target, mask, states) :              
        # Apply mask to target and prediction
        masked_target = target[mask]
        masked_pred = pred[mask]

        if len(masked_pred) > 0 :
            # Compute error metrics
            error = masked_pred - masked_target
            states.sum_error += torch.sum(error)      
            states.sum_squared_error += torch.sum(error ** 2)
            states.sum_absolute_error += torch.sum(torch.abs(error))
            
            # Compute relative error metrics
            mask_min_height = masked_target >= self.min_height_nMAE
            states.sum_relative_error += torch.sum(torch.abs(masked_pred[mask_min_height] - masked_target[mask_min_height])/ (1 + masked_target[mask_min_height]))

            # Compute tree cover metrics
            tree_cover_target = target >= self.tree_cover_threshold
            tree_cover_pred = pred >= self.tree_cover_threshold
            states.intersection += torch.logical_and(tree_cover_pred, tree_cover_target).sum()
            states.union += torch.logical_or(tree_cover_pred, tree_cover_target).sum()

            # Compute number of values for each mask
            states.nb_values += mask.sum() 
            states.nb_values_min_height += mask_min_height.sum()

    def epoch_compute(self, states):
        final_results = {}    
        # If there are values, compute metrics
        if states.nb_values > 0 :
            # ME - Mean Error
            final_results["ME"] = (states.sum_error / states.nb_values).to(torch.float32)

            # MAE - Mean Absolute Error
            final_results["MAE"] = (states.sum_absolute_error / states.nb_values).to(torch.float32)

            # RMSE - Root Mean Square Error
            final_results["RMSE"] = torch.sqrt(states.sum_squared_error / states.nb_values).to(torch.float32)

            final_results["nMAE"] = (states.sum_relative_error / states.nb_values_min_height).to(torch.float32)

            # TreeCov - Treecover IoU
            final_results["TreeCov"] = (states.intersection / states.union).to(torch.float32) if states.union > 0 else torch.tensor(torch.nan)

            #nb_values - Number of values
            final_results["nb_values"] = states.nb_values.to(torch.float32)
        else :
            # If no values, set all metrics to NaN
            final_results["ME"] = torch.tensor(torch.nan)
            final_results["MAE"] = torch.tensor(torch.nan)
            final_results["RMSE"] = torch.tensor(torch.nan)
            final_results["nMAE"] = torch.tensor(torch.nan)
            final_results["TreeCov"] = torch.tensor(torch.nan)
            final_results["nb_values"] = torch.tensor(0)
        
        return final_results

