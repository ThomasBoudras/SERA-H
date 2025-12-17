from typing import Any
from pathlib import Path
import torch
from lightning import LightningModule
import numpy as np
import torch.nn.functional as F

class Module(LightningModule):
    """
    LightningModule for the SERA-H model.
    
    This module combines a super-resolution model (EDSR) and a regression model 
    (typically UTAE or U-Net) to predict canopy height maps from low-resolution 
    satellite imagery (Sentinel-1 & Sentinel-2) under the supervision of high-resolution 
    LiDAR data (ALS).

    Args:
        super_resolution_model (nn.Module): The super-resolution network (e.g., EDSR).
        regression_model (nn.Module): The regression network (e.g., UTAE).
        loss (nn.Module): The loss function used for training.
        train_metrics (MetricCollection): Metrics to track during training.
        val_metrics (MetricCollection): Metrics to track during validation.
        test_metrics (MetricCollection): Metrics to track during testing.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        optimizer (torch.optim.Optimizer): Optimizer.
        predictions_save_dir (str): Directory where predictions will be saved.
    """
    def __init__(
        self,
        super_resolution_model,
        regression_model,
        loss,
        train_metrics,
        val_metrics,
        test_metrics,
        scheduler,
        optimizer,
        predictions_save_dir,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["super_resolution_model", "regression_model", "loss", "train_metrics", "val_metrics", "test_metrics"]) # We do not save the models in the hyperparameters of nn.Module to avoid duplication

        self.super_resolution_model = super_resolution_model
        self.regression_model = regression_model
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictions_save_dir = Path(predictions_save_dir).resolve()
    
    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor, meta_data):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing satellite time series.
                              Shape: (Batch, Time, Channels, Height, Width)
            meta_data (dict): Metadata associated with the input images (e.g., acquisition dates).

        Returns:
            torch.Tensor: Predicted canopy height map.
                          Shape: (Batch, 1, Height_HR, Width_HR)
        """
        if self.super_resolution_model is not None :
            x = self.super_resolution_model(x, meta_data)
        preds = self.regression_model(x, meta_data)
        return preds

    def step(self, batch: Any, stage, metrics_function):
        inputs, targets, meta_data = batch
        preds = self.forward(inputs, meta_data)
        
        loss = self.loss(preds, targets)

        self.log(
            name= f"{stage}/loss",
            value=loss, 
            on_step=True,
            on_epoch=True, 
            prog_bar=False,
            sync_dist= True
        )
    
        if metrics_function :
            metrics_function.update(preds, targets, meta_data)
        return loss, preds, targets
    

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="train", metrics_function=self.train_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="val", metrics_function=self.val_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="test", metrics_function=self.test_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def final_step(self, stage, metrics_function):
        if stage == "val":
            # Log the patience countdown for EarlyStopping callback if present
            for callback in self.trainer.callbacks:
                # EarlyStopping patience countdown
                if hasattr(callback, "patience") and hasattr(callback, "wait_count"):
                    remaining = callback.patience - callback.wait_count
                    self.log(
                        f"early_stopping_patience_remaining",
                        remaining
                    )

            # Log the remaining patience for the scheduler (ReduceLROnPlateau) if present
            if hasattr(self.trainer, "lr_scheduler_configs"):
                for sched_cfg in self.trainer.lr_scheduler_configs:
                    scheduler = sched_cfg.scheduler
                    # Check for ReduceLROnPlateau attributes
                    if hasattr(scheduler, "patience") and hasattr(scheduler, "num_bad_epochs"):
                        remaining = scheduler.patience - scheduler.num_bad_epochs
                        self.log(
                            f"scheduler_patience_remaining",
                            remaining
                        )


        if metrics_function is not None :
            metrics = metrics_function.compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{stage}/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    )
            metrics_function.reset()   

    def on_train_epoch_end(self):
        self.final_step("train", self.train_metrics)
    
    def on_validation_epoch_end(self):
        self.final_step("val", self.val_metrics)

    def on_test_epoch_end(self):
        self.final_step("test", self.test_metrics)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # At predict time, there are (normally) only inputs, no targets
        inputs, _, meta_data = batch
        preds = self(inputs, meta_data)

        crop_size = int(preds.shape[-1]/12) 
        preds[...,:crop_size, :] = torch.nan 
        preds[...,-crop_size:, :] = torch.nan  
        preds[...,:, :crop_size] = torch.nan  
        preds[...,:, -crop_size:] = torch.nan  
        
        bounds = meta_data["bounds"]
        
        if self.predictions_save_dir is not None:
            (self.predictions_save_dir / f"rank_{self.global_rank}" / "preds").mkdir(parents=True, exist_ok=True)
            (self.predictions_save_dir / f"rank_{self.global_rank}" / "bounds").mkdir(parents=True, exist_ok=True)
            np.save(
                self.predictions_save_dir / f"rank_{self.global_rank}" / "preds" / f"batch_{batch_idx}.npy",
                preds.cpu().numpy().astype(np.float32),
            )
            np.save(
                self.predictions_save_dir / f"rank_{self.global_rank}" / "bounds" / f"batch_{batch_idx}.npy",
                bounds.cpu().numpy().astype(np.float32),
            )
        else:
            raise Exception("Please give a name for the prediction dir ")