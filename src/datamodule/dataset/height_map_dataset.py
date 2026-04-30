from typing import Any
import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision.transforms import v2
from safetensors.torch import save_file
from safetensors import safe_open
from pathlib import Path

from src.datamodule.datamodule_utils import get_random_tensor_crop
from src.global_utils import subdivide_bounds_gdf, expand_bounds_gdf, get_logger

log = get_logger(__name__)

class heightMapDataset(Dataset):
    def __init__(
        self,
        resolution_input,
        resolution_target,
        patch_size_input,
        patch_size_target,
        gdf_path,
        data_augmentation, 
        get_inputs,
        get_targets,
        save_dir,
        batch_size_save,
        mode,
        stage,
    ):
        self.resolution_input = resolution_input
        self.resolution_target = resolution_target
        self.patch_size_input = patch_size_input
        self.patch_size_target = patch_size_target
        self.patch_size_real = patch_size_input*resolution_input 
        self.data_augmentation = data_augmentation
        
        self.gdf_path = gdf_path
        self.get_inputs = get_inputs
        self.get_targets = get_targets
        self.save_dir = Path(save_dir)
        self.batch_size_save = batch_size_save
        self.mode = mode        
        self.stage = stage
        self.gdf = self.prepare_gdf()

        self.transform_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transform_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.set_mode(self.mode)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_mode(self, mode):
        self.mode = mode
        self.get_item = self._load_getitem if mode == "load" else self._save_getitem
        self.collate_fn = self._collate_fn_load if mode == "load" else self._collate_fn_save


    def __len__(self):
        return len(self.gdf)
    
    def prepare_gdf(self):
        gdf = gpd.read_file(self.gdf_path)

        if "predict" not in self.stage : #we predict on all patches of the gdf
            gdf = gdf[gdf['split'] == self.stage ].reset_index(drop=True)   
        if "predict" in self.stage and "test" in self.stage:
            gdf = gdf[gdf['split'] == "test" ].reset_index(drop=True)   

        gdf = self.get_inputs.prepare_gdf_for_inputs(gdf)

        gdf["initial_index"] = gdf.index

        if "predict" in self.stage :
            margin_size = int(self.patch_size_real / 12) # By convention, we use a margin of 1/12 of the patch size
            
            if "subdivide" in self.stage and not self.mode == "save": # case where we want to subdivide the gdf into subpatches
                gdf = subdivide_bounds_gdf(gdf, patch_size=self.patch_size_real, margin_size=margin_size, resolution=self.resolution_input)
            else: # case where we want to expand the bounds of the gdf with the margin 
                gdf = expand_bounds_gdf(gdf, margin_size=margin_size, resolution=self.resolution_input)
        
        if self.mode == "load" and self.stage == "train" and "nb_repetitions" in gdf.columns:
            gdf = gdf.loc[gdf.index.repeat(gdf["nb_repetitions"])].reset_index(drop=True)
        
        log.info(f"Number of {self.stage} samples : {len(gdf)}")
        return gdf

    def update_moments_transforms(self, input_mean, input_std, transform_input):
        self.transform_input = transform_input
        
        self.get_inputs.mean = input_mean
        self.get_inputs.std = input_std
        
    def _dict_to_batch(self, dict_batch):
        inputs = [item[0] for item in dict_batch]
        targets = [item[1] for item in dict_batch]
        metadata = [item[2] for item in dict_batch]
        batch_input = torch.stack(inputs, dim=0)
        batch_target = torch.stack(targets, dim=0)

        batch_metadata = {key: torch.stack([d[key] for d in metadata], dim=0) for key in metadata[0]}

        return batch_input, batch_target, batch_metadata
        
    def _save_getitem(self, ix):

        idx_batch = ix // self.batch_size_save
        if (self.save_dir / f"batch_{idx_batch}.safetensors").exists():
            return None, None, None
        
        row_gdf = self.gdf.loc[ix]
        bounds = list(row_gdf["geometry"].bounds)

        #Bottom Left crop, else we save the full patch
        if not self.data_augmentation and "predict" not in self.stage: 
            bounds[2], bounds[3] = bounds[0] + self.patch_size_real, bounds[1] + self.patch_size_real
        
        inputs, metadata_inputs = self.get_inputs(bounds, row_gdf, self.transform_input)
        targets, metadata_targets = self.get_targets(bounds, row_gdf, self.transform_target)  
        
        metadata = {}
        for key, value in metadata_inputs.items() :
            metadata[key] = value
        for key, value in metadata_targets.items() :
            metadata[key] = value

        metadata["row_idx"] = torch.tensor([ix])
        metadata["input_shape"] = torch.tensor(inputs.shape)

        return inputs, targets, metadata

    def _collate_fn_save(self, batch):

        if batch[0][0] is None:
            return
        
        batch_input, batch_target, batch_metadata = self._dict_to_batch(batch)
        ix = list(batch_metadata["row_idx"])
        
        #Check if the indices are consecutive
        for i in range(1, len(ix)):
            if ix[i] != ix[i - 1] + 1:
                raise ValueError(f"Batch indices not consecutive: {ix[i-1]} followed by {ix[i]}")

        idx_batch = ix[-1].item()//self.batch_size_save

        tensors_dict = {
            "inputs": batch_input,
            "targets": batch_target
        }
        for k, v in batch_metadata.items():
            tensors_dict[f"meta_{k}"] = v

        # Save the tensors to a temporary file and then rename it to the final path
        tmp_path = self.save_dir / f"batch_{idx_batch}.safetensors.tmp"
        final_path = self.save_dir / f"batch_{idx_batch}.safetensors"

        # Remove the temporary file if it exists
        if tmp_path.exists():
            tmp_path.unlink()
        
        # Save the tensors to the temporary file
        save_file(tensors_dict, tmp_path)

        # Rename the temporary file to the final path
        tmp_path.rename(final_path)


    def _load_getitem(self, ix):

        row_gdf = self.gdf.loc[ix]
        idx_save = row_gdf["initial_index"] 

        ix_batch = idx_save // self.batch_size_save
        path = self.save_dir / f"batch_{ix_batch}.safetensors"
        
        ix_in_batch = idx_save % self.batch_size_save
        
        bounds = list(row_gdf["geometry"].bounds)

        metadata = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            inputs_slice_obj = f.get_slice("inputs")
            targets_slice_obj = f.get_slice("targets")
            
            for k in f.keys():
                if k.startswith("meta_"):
                    key_name = k.replace("meta_", "", 1)
                    metadata[key_name] = f.get_slice(k)[ix_in_batch]

            # Case only for train, test, and val 
            if "predict" not in self.stage:
                if self.data_augmentation : # case where we do data augmentation in training mode         
                    # get random crop
                    input_shape = metadata["input_shape"]
                    crop_tensor_input = get_random_tensor_crop(input_shape, self.patch_size_input)
                    factor_inputs_to_targets = self.patch_size_target / self.patch_size_input
                    crop_tensor_target = [int(position * factor_inputs_to_targets) for position in crop_tensor_input]
                    if len(input_shape) == 3: #shape CHW
                        inputs = inputs_slice_obj[ix_in_batch, :, crop_tensor_input[1]:crop_tensor_input[3], crop_tensor_input[0]:crop_tensor_input[2]]
                    elif len(input_shape) == 4: #shape TCHW
                        inputs = inputs_slice_obj[ix_in_batch, :, :, crop_tensor_input[1]:crop_tensor_input[3], crop_tensor_input[0]:crop_tensor_input[2]]
                    else:
                        raise ValueError(f"Input shape {input_shape} not supported")

                    targets = targets_slice_obj[ix_in_batch, :, crop_tensor_target[1]:crop_tensor_target[3], crop_tensor_target[0]:crop_tensor_target[2]]

                    # get random rotation
                    rotation = np.random.randint(0,4)
                    if rotation > 0 :
                        inputs = torch.rot90(inputs, k=rotation, dims=(-1, -2))
                        targets = torch.rot90(targets, k=rotation, dims=(-1, -2))
                    
                    # get random mirror
                    mirror = np.random.randint(0,3)
                    if mirror > 0 :
                        inputs = torch.flip(inputs, dims=[-mirror])  
                        targets = torch.flip(targets, dims=[-mirror]) 
                
                    # update the bounds to the crop
                    bounds[0] += crop_tensor_input[0] * self.resolution_input
                    bounds[1] += crop_tensor_input[1] * self.resolution_input
                    bounds[2] = bounds[0] + self.patch_size_real
                    bounds[3] = bounds[1] + self.patch_size_real
                else : 
                    bounds[2] = bounds[0] + self.patch_size_real
                    bounds[3] = bounds[1] + self.patch_size_real
                    inputs = inputs_slice_obj[ix_in_batch] # already cropped in bottom left corner, just need to get the slice
                    targets = targets_slice_obj[ix_in_batch]
            
            # Case predict on subdivided patches
            elif "subdivide" in self.stage: 
                
                #get the crop tensor for the input
                input_shape = metadata["input_shape"]
                global_bounds = row_gdf["patch_bounds"]
                offset_x = int((bounds[0] - global_bounds[0]) / self.resolution_input)
                offset_y = int((global_bounds[3] - bounds[3]) / self.resolution_input)
                crop_tensor_input = [offset_x, offset_y, offset_x + self.patch_size_input, offset_y + self.patch_size_input]

                if len(input_shape) == 3: #shape CHW
                     inputs = inputs_slice_obj[ix_in_batch, :, crop_tensor_input[1]:crop_tensor_input[3], crop_tensor_input[0]:crop_tensor_input[2]]
                elif len(input_shape) == 4: #shape TCHW
                    inputs = inputs_slice_obj[ix_in_batch, :, :, crop_tensor_input[1]:crop_tensor_input[3], crop_tensor_input[0]:crop_tensor_input[2]]
                else:
                    raise ValueError(f"Input shape {input_shape} not supported")

                targets = torch.tensor(torch.nan)

            # Case predict on full patches
            else : 
                inputs = inputs_slice_obj[ix_in_batch]
                targets = targets_slice_obj[ix_in_batch]
                
        metadata["bounds"] = torch.tensor(bounds)
        return inputs, targets, metadata

    def _collate_fn_load(self, batch):
        batch_input, batch_target, batch_metadata = self._dict_to_batch(batch)
        return batch_input, batch_target, batch_metadata
    
    def __getitem__(self, ix):
        return self.get_item(ix)

    def __collate_fn__(self, batch):
        return self.collate_fn(batch)
    
