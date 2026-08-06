"""PyTorch Lightning callback that logs validation/test/train image samples to TensorBoard."""

from typing import Any, Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchvision.utils import make_grid

from src.callbacks.log_images_utils import get_tensorboard_logger


class LogImages(Callback):
    """Logs a batch of validation samples and their predictions to TensorBoard.

    Args:
        num_samples (int): Number of samples to log.
        freq_train: Frequency of logging during training, if None, no logging during training.
        prepare_images: class to prepare images for logging. To be adapted to the nature of the images.
        log_inputs: class to log input images. To be adapted to the nature of the images.
        log_predictions: class to log prediction images, if None, predictions are logged with a default grid.
    """

    def __init__(
        self,
        num_samples: int,
        freq_train: Optional[int],
        prepare_images: Any,
        log_inputs: Any,
        log_predictions: Optional[Any],
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.freq_train = freq_train
        self.prepare_images = prepare_images
        self.log_inputs = log_inputs
        self.log_predictions = log_predictions

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Logs validation samples at the end of the validation epoch.

        Skipped during Lightning's sanity check to avoid logging on partial/dummy data.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            pl_module (LightningModule): The Lightning module being trained.
        """
        # Only save images if not in sanity check, using trainer.running_sanity_check
        if not trainer.sanity_checking:
            self._save_images(trainer, pl_module, trainer.datamodule.val_dataset, stage="val")

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Logs test samples at the end of testing.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            pl_module (LightningModule): The Lightning module being tested.
        """
        self._save_images(trainer, pl_module, trainer.datamodule.test_dataset, stage="test")

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Logs training samples every `freq_train` batches.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            pl_module (LightningModule): The Lightning module being trained.
            outputs (Any): Outputs of the training step for the current batch.
            batch (Any): The current training batch.
            batch_idx (int): Index of the current training batch.
        """
        if self.freq_train is not None and batch_idx % self.freq_train == 0:
            self._save_images(
                trainer,
                pl_module,
                trainer.datamodule.train_dataset,
                stage=f"train_step_{batch_idx}",
            )

    def _save_images(
        self, trainer: Trainer, pl_module: LightningModule, dataset: Any, stage: str
    ) -> None:
        """Builds and logs input, target, and prediction image grids to TensorBoard.

        Draws `num_samples` evenly spaced samples from `dataset`, runs them through
        `pl_module`, prepares the resulting images, and logs them via `log_inputs`
        and either `log_predictions` (if set) or a default image grid.

        Args:
            trainer (Trainer): The Lightning trainer instance, used to access the logger.
            pl_module (LightningModule): The Lightning module used to compute predictions.
            dataset (Any): Dataset to sample images from.
            stage (str): Name of the stage (e.g. "val", "test", "train_step_<idx>") used
                to namespace the logged images.
        """
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        self.device = pl_module.device

        inputs_images = []
        target_images = []
        pred_images = []
        metadata_images = []
        for i in range(self.num_samples):
            factor = len(dataset) // self.num_samples
            idx = i * factor
            # Get the i-th sample from the dataset
            input_i, target_i, metadata_i = dataset[idx]

            # Convert input and target to tensors and move to the correct device
            input_tensor = input_i.to(device=pl_module.device)
            target_tensor = target_i.to(device=pl_module.device)
            metadata_i = {
                key: metadata_i[key].unsqueeze(0).to(device=pl_module.device) for key in metadata_i
            }

            # Get prediction for this sample
            pred_tensor = (
                pl_module(input_tensor.unsqueeze(0), target_tensor.unsqueeze(0), metadata_i)
                .squeeze(0)
                .to(device=pl_module.device)
            )

            # Prepare images for logging
            pred_image, target_image = self.prepare_images(pred_tensor, target_tensor)

            inputs_images.append(input_tensor)
            target_images.append(target_image)
            metadata_images.append(metadata_i)
            pred_images.append(pred_image)

        # log input images
        self.log_inputs(inputs_images, experiment, stage)

        curr_epoch = int(trainer.current_epoch)
        if self.log_predictions is not None:
            self.log_predictions(pred_images, curr_epoch, experiment, stage)
        else:
            pred_images_grid = make_grid(pred_images)
            experiment.add_image(
                f"predicted_images/{stage}", pred_images_grid, global_step=curr_epoch
            )

        # log target images
        target_images_grid = make_grid(target_images)
        experiment.add_image(
            f"target_images/{stage}", target_images_grid, global_step=0
        )  # As the target are the same each epoch, we dont specify the epoch
