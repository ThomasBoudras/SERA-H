import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision.transforms import v2

from src.datamodule.datamodule_utils import generate_random_bounds, expand_gdf
from src.global_utils import get_logger

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
        self.stage = stage
        self.gdf = self.prepare_gdf()

        self.transform_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transform_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.gdf)
    
    def prepare_gdf(self):
        gdf = gpd.read_file(self.gdf_path)

        if self.stage != "predict" : #we predict on all patches of the gdf
            gdf = gdf[gdf['split'] == self.stage ].reset_index(drop=True)   

        gdf = self.get_inputs.prepare_gdf_for_inputs(gdf)

        if self.stage == "predict" :
            # In prediction mode, we want to generate predictions over the entire geometry,
            # while respecting the maximum patch size the model can process.
            # Therefore, we split each geometry into sub-patches, adding a margin to each
            # sub-patch to reduce edge effects during prediction.

            margin_size = int(self.patch_size_real / 12)
            # If we used a margin of 1/6 of the patch size, the patches would completely overlap,
            # which is a bit excessive. Instead, by convention, we use a margin of 1/12 of the patch size
            # to provide some overlap for edge effects, but not a full overlap.

            gdf = expand_gdf(gdf, patch_size=self.patch_size_real, margin_size=margin_size, resolution=self.resolution_input)
        
        log.info(f"Number of {self.stage} samples : {len(gdf)}")
        return gdf

    def update_moments_transforms(self, input_mean, input_std, transform_input):
        self.transform_input = transform_input
        
        self.get_inputs.mean = input_mean
        self.get_inputs.std = input_std
        

    def custom_collate_fn(self, batch):
        inputs = [item[0] for item in batch]  
        targets = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(inputs, dim=0)
        batch_target = torch.stack(targets, dim=0)

        batch_meta_data = {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data
    
    def __getitem__(self, ix):
        row_gdf = self.gdf.loc[ix]
        bounds = list(row_gdf["geometry"].bounds)

        #Random Crop
        if self.data_augmentation and self.stage != "predict" : 
            bounds = generate_random_bounds(bounds=bounds, patch_size=self.patch_size_real, resolution=self.resolution_input)
            mirror = np.random.randint(0,3)
            rotation = np.random.randint(0,4)

        #Bottom Left crop
        else : 
            bounds[2], bounds[3] = bounds[0] + self.patch_size_real, bounds[1] + self.patch_size_real
            mirror = 0
            rotation = 0
        
        inputs, meta_data_inputs = self.get_inputs(bounds, row_gdf, self.transform_input)
        targets, meta_data_targets = self.get_targets(bounds, row_gdf, self.transform_target)

        if rotation > 0 :
            inputs = torch.rot90(inputs, k=rotation, dims=(-1, -2))
            targets = torch.rot90(targets, k=rotation, dims=(-1, -2))
        
        if mirror > 0 :
            inputs = torch.flip(inputs, dims=[-mirror])  
            targets = torch.flip(targets, dims=[-mirror])  
        
        meta_data = {}
        for key, value in meta_data_inputs.items() :
            meta_data[key] = value
        for key, value in meta_data_targets.items() :
            meta_data[key] = value

        meta_data["bounds"] = torch.tensor(bounds)

        return inputs, targets, meta_data
    
