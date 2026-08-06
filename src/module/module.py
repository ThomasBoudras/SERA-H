from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningModule


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
        super_resolution_model: Optional[torch.nn.Module],
        regression_model: torch.nn.Module,
        loss: torch.nn.Module,
        train_metrics: Any,
        val_metrics: Any,
        test_metrics: Any,
        scheduler: Any,
        optimizer: Any,
        predictions_save_dir: str,
    ) -> None:
        """Initialize the module and store its submodules, metrics, and optimization settings.

        Args:
            super_resolution_model (Optional[torch.nn.Module]): The super-resolution network
                (e.g., EDSR), or None if no super-resolution step is used.
            regression_model (torch.nn.Module): The regression network (e.g., UTAE or U-Net).
            loss (torch.nn.Module): The loss function used for training.
            train_metrics (Any): Metrics (e.g. torchmetrics MetricCollection) to track during training.
            val_metrics (Any): Metrics (e.g. torchmetrics MetricCollection) to track during validation.
            test_metrics (Any): Metrics (e.g. torchmetrics MetricCollection) to track during testing.
            scheduler (Any): Learning rate scheduler factory (partial), or None.
            optimizer (Any): Optimizer factory (partial).
            predictions_save_dir (str): Directory where predictions will be saved.
        """
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "super_resolution_model",
                "regression_model",
                "loss",
                "train_metrics",
                "val_metrics",
                "test_metrics",
            ]
        )  # We do not save the models in the hyperparameters of nn.Module to avoid duplication

        self.super_resolution_model = super_resolution_model
        self.regression_model = regression_model
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictions_save_dir = Path(predictions_save_dir).resolve()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Instantiate the optimizer and (optionally) the learning rate scheduler.

        Returns:
            Dict[str, Any]: A dict with the "optimizer" key and, if a scheduler is configured,
                an "lr_scheduler" key describing the scheduler configuration (monitored on
                "val/loss", stepped every epoch).
        """
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

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor(s), typically satellite time series or images.
            targets (torch.Tensor): Target tensors (e.g., ground truth height maps), may be required by some models.
            metadata (Dict[str, Any]): Metadata associated with inputs (e.g., acquisition dates, auxiliary info).

        Returns:
            torch.Tensor: Predicted outputs (e.g., canopy height map or super-resolved image).
        """

        if self.super_resolution_model is not None:
            inputs = self.super_resolution_model(inputs, targets, metadata)
        preds = self.regression_model(inputs, targets, metadata)
        return preds

    def step(
        self, batch: Any, stage: str, metrics_function: Optional[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass, compute the loss, log it, and update the given metrics.

        Args:
            batch (Any): Tuple of (inputs, targets, metadata) produced by the dataloader.
            stage (str): Name of the current stage (e.g. "train", "val", "test"), used for logging.
            metrics_function (Optional[Any]): Metric collection to update with (preds, targets, metadata),
                or None to skip metric updates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The loss, predictions, and targets.
        """
        inputs, targets, metadata = batch
        preds = self.forward(inputs, targets, metadata)

        loss = self.loss(preds, targets, metadata)

        self.log(
            name=f"{stage}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if metrics_function:
            metrics_function.update(preds, targets, metadata)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Run a single training step.

        Args:
            batch (Any): Tuple of (inputs, targets, metadata) produced by the dataloader.
            batch_idx (int): Index of the batch within the current epoch.

        Returns:
            Dict[str, torch.Tensor]: Dict with the "loss", "preds", and "targets" tensors.
        """
        loss, preds, targets = self.step(
            batch=batch, stage="train", metrics_function=self.train_metrics
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Run a single validation step.

        Args:
            batch (Any): Tuple of (inputs, targets, metadata) produced by the dataloader.
            batch_idx (int): Index of the batch within the current epoch.

        Returns:
            Dict[str, torch.Tensor]: Dict with the "loss", "preds", and "targets" tensors.
        """
        loss, preds, targets = self.step(
            batch=batch, stage="val", metrics_function=self.val_metrics
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Run a single test step.

        Args:
            batch (Any): Tuple of (inputs, targets, metadata) produced by the dataloader.
            batch_idx (int): Index of the batch within the current epoch.

        Returns:
            Dict[str, torch.Tensor]: Dict with the "loss", "preds", and "targets" tensors.
        """
        loss, preds, targets = self.step(
            batch=batch, stage="test", metrics_function=self.test_metrics
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def final_step(self, stage: str, metrics_function: Optional[Any]) -> None:
        """Log epoch-end information: scheduler/early-stopping patience and aggregated metrics.

        Args:
            stage (str): Name of the current stage (e.g. "train", "val", "test"), used for logging.
            metrics_function (Optional[Any]): Metric collection to compute, log, and reset,
                or None to skip metric logging.
        """
        if stage == "val":
            # Log the patience countdown for EarlyStopping callback if present
            for callback in self.trainer.callbacks:
                # EarlyStopping patience countdown
                if hasattr(callback, "patience") and hasattr(callback, "wait_count"):
                    remaining = callback.patience - callback.wait_count
                    self.log("early_stopping_patience_remaining", remaining)

            # Log the remaining patience for the scheduler (ReduceLROnPlateau) if present
            if hasattr(self.trainer, "lr_scheduler_configs"):
                for sched_cfg in self.trainer.lr_scheduler_configs:
                    scheduler = sched_cfg.scheduler
                    # Check for ReduceLROnPlateau attributes
                    if hasattr(scheduler, "patience") and hasattr(scheduler, "num_bad_epochs"):
                        remaining = scheduler.patience - scheduler.num_bad_epochs
                        self.log("scheduler_patience_remaining", remaining)

        if metrics_function is not None:
            metrics = metrics_function.compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{stage}/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                )
            metrics_function.reset()

    def on_train_epoch_end(self) -> None:
        """Log aggregated training metrics at the end of the epoch."""
        self.final_step("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        """Log aggregated validation metrics at the end of the epoch."""
        self.final_step("val", self.val_metrics)

    def on_test_epoch_end(self) -> None:
        """Log aggregated test metrics at the end of the epoch."""
        self.final_step("test", self.test_metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Run inference on a batch, mask border artifacts, and save predictions/bounds to disk.

        Args:
            batch (Any): Tuple of (inputs, targets, metadata) produced by the dataloader.
                At predict time, there are (normally) only inputs, no targets.
            batch_idx (int): Index of the batch within the current epoch.
            dataloader_idx (int): Index of the dataloader when multiple dataloaders are used.

        Raises:
            Exception: If `predictions_save_dir` was not set.
        """
        inputs, targets, metadata = batch
        preds = self(inputs, targets, metadata)

        crop_size = int(preds.shape[-1] / 12)
        preds[..., :crop_size, :] = torch.nan
        preds[..., -crop_size:, :] = torch.nan
        preds[..., :, :crop_size] = torch.nan
        preds[..., :, -crop_size:] = torch.nan

        bounds = metadata["bounds"]

        if self.predictions_save_dir is not None:
            (self.predictions_save_dir / f"rank_{self.global_rank}" / "preds").mkdir(
                parents=True, exist_ok=True
            )
            (self.predictions_save_dir / f"rank_{self.global_rank}" / "bounds").mkdir(
                parents=True, exist_ok=True
            )
            np.save(
                self.predictions_save_dir
                / f"rank_{self.global_rank}"
                / "preds"
                / f"batch_{batch_idx}.npy",
                preds.cpu().numpy().astype(np.float32),
            )
            np.save(
                self.predictions_save_dir
                / f"rank_{self.global_rank}"
                / "bounds"
                / f"batch_{batch_idx}.npy",
                bounds.cpu().numpy().astype(np.float32),
            )
        else:
            raise Exception("Please give a name for the prediction dir ")
