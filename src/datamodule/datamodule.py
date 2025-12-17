import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from pathlib import Path

from src.datamodule.datamodule_utils import SubsetSampler
from src import global_utils as utils

log = utils.get_logger(__name__)

class Datamodule(L.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        max_n_inputs_for_moments_computation: int,
        max_n_inputs_per_epoch: int,
        normalization_save_path,
        train_dataset,
        val_dataset,
        test_dataset,
        predict_dataset,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        self.normalization_save_path = Path(self.hparams.normalization_save_path).resolve()
        
        if not (self.normalization_save_path / "mean.npy").exists():
            log.info("Computing normalization moments...")
            if not self.normalization_save_path.exists():
                self.normalization_save_path.mkdir(parents=True, exist_ok=True)

            # Instantiate a temporary dataset for calculation
            temp_train_dataset = self.hparams.train_dataset
            
            def get_mean_std(sample, axis):
                x, y , meta_data = sample
                x = x.squeeze().to(torch.float32)  
                if len(x.shape) == 4:
                    x = x[0]
                if "inputs_seperation" in meta_data:
                    x = x[:meta_data["inputs_seperation"], ...] 
                x[~torch.isfinite(x)] = 0
                return x.mean(axis=axis), x.std(axis=axis)

            sampled_indices = np.random.choice(
                len(temp_train_dataset),
                min(len(temp_train_dataset), self.hparams.max_n_inputs_for_moments_computation),
                replace=False
            )

            mean_per_image, std_per_image = zip(
                *[
                    get_mean_std(temp_train_dataset[ix], axis=(1, 2))
                    for ix in tqdm(sampled_indices, desc="Computing mean and std of inputs")
                ]
            )

            input_mean = np.mean(np.array(mean_per_image), axis=0)
            input_std = np.mean(np.array(std_per_image), axis=0)

            np.save(self.normalization_save_path / "mean.npy", input_mean)
            np.save(self.normalization_save_path / "std.npy", input_std)

    def setup(self, stage: str):
        # This method is called on every rank.
        self.normalization_save_path = Path(self.hparams.normalization_save_path).resolve()
        self.input_mean = np.load(self.normalization_save_path / "mean.npy")
        self.input_std = np.load(self.normalization_save_path / "std.npy")
        
        log.info(f"## values mean : {self.input_mean}  ##")
        log.info(f"## values std : {self.input_std}  ##")

        # Once we have moment we add normalization transform to dataset
        transform_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=self.input_mean, std=self.input_std)])
        self.train_dataset = self.hparams.train_dataset
        self.val_dataset = self.hparams.val_dataset
        self.test_dataset = self.hparams.test_dataset
        self.predict_dataset = self.hparams.predict_dataset
        
        if self.train_dataset is not None :
            self.train_dataset.update_moments_transforms(self.input_mean, self.input_std, transform_input)
        if self.val_dataset is not None :
            self.val_dataset.update_moments_transforms(self.input_mean, self.input_std, transform_input)
        if self.test_dataset is not None :
            self.test_dataset.update_moments_transforms(self.input_mean, self.input_std, transform_input)
        if self.predict_dataset is not None :
            self.predict_dataset.update_moments_transforms(self.input_mean, self.input_std, transform_input)
        

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.hparams.max_n_inputs_per_epoch:
            sampler = SubsetSampler(self.train_dataset, self.hparams.max_n_inputs_per_epoch, True)
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            collate_fn=self.train_dataset.custom_collate_fn if hasattr(self.train_dataset, 'custom_collate_fn') else None,
            timeout=600,
        )

    def val_dataloader(self):
        sampler = None
        if self.hparams.max_n_inputs_per_epoch:
            sampler = SubsetSampler(self.val_dataset, self.hparams.max_n_inputs_per_epoch, False)
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            collate_fn=self.val_dataset.custom_collate_fn if hasattr(self.val_dataset, 'custom_collate_fn') else None,
            timeout=600,
        )

    def test_dataloader(self):
        sampler = None
        if self.hparams.max_n_inputs_per_epoch:
            sampler = SubsetSampler(self.test_dataset, self.hparams.max_n_inputs_per_epoch, False)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            collate_fn=self.test_dataset.custom_collate_fn if hasattr(self.test_dataset, 'custom_collate_fn') else None,
            timeout=600,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            collate_fn=self.predict_dataset.custom_collate_fn if hasattr(self.predict_dataset, 'custom_collate_fn') else None,
            timeout=600,
        )

    

