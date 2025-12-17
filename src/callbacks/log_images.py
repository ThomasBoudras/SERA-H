import torch
from torchvision.utils import make_grid
from src.callbacks.log_images_utils import get_tensorboard_logger
from lightning.pytorch import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class LogImages(Callback):
    """
    Logs a batch of validation samples and their predictions to TensorBoard.

    Args:
        num_samples (int): Number of samples to log.
        freq_train: Frequency of logging during training, if None, no logging during training.
        prepare_images: class to prepare images for logging. To be adapted to the nature of the images.
        log_inputs: class to log input images. To be adapted to the nature of the images.
    """

    def __init__(self, num_samples: int, freq_train, prepare_images, log_inputs):
        super().__init__()
        self.num_samples = num_samples
        self.freq_train = freq_train
        self.prepare_images = prepare_images
        self.log_inputs = log_inputs

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # Only save images if not in sanity check, using trainer.running_sanity_check
        if not trainer.sanity_checking :
            self._save_images(trainer, pl_module, trainer.datamodule.val_dataset, stage="val")

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        self._save_images(trainer, pl_module, trainer.datamodule.test_dataset, stage="test")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq_train is not None and batch_idx % self.freq_train == 0:
            self._save_images(trainer, pl_module, trainer.datamodule.train_dataset, stage=f"train_step_{batch_idx}")

    def _save_images(self, trainer, pl_module, dataset, stage):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        self.device = pl_module.device

        inputs_images = []
        target_images = []
        pred_images = []

        for i in range(self.num_samples):
            # Get the i-th sample from the dataset
            input_i, target_i, meta_data_i = dataset[i]

            # Convert input and target to tensors and move to the correct device
            input_tensor = input_i.to(device=pl_module.device)
            target_tensor = target_i.to(device=pl_module.device)
            meta_data_i = {key : meta_data_i[key].unsqueeze(0) for key in meta_data_i}

            # Get prediction for this sample
            pred_tensor = pl_module(input_tensor.unsqueeze(0), meta_data_i).squeeze(0).to(device=pl_module.device)

            # Prepare images for logging
            pred_image, target_image = self.prepare_images(pred_tensor, target_tensor)

            inputs_images.append(input_tensor)
            target_images.append(target_image)
            pred_images.append(pred_image)
        
        # log input images
        self.log_inputs(inputs_images, experiment, stage)

        # log target and predicted images
        curr_epoch = int(trainer.current_epoch)

        target_images_grid = make_grid(target_images)
        pred_images_grid = make_grid(pred_images)
        # TensorBoard expects tensors (C, H, W) or (N, C, H, W)
        experiment.add_image(f"target_images/{stage}", target_images_grid, global_step=0) # As the target are the same each epoch, we dont specify the epoch
        experiment.add_image(f"predicted_images/{stage}/epoch_{curr_epoch}", pred_images_grid, global_step=curr_epoch)
