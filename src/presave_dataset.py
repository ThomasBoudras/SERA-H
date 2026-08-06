"""Pre-compute and cache dataset batches to disk ahead of training/validation/test/predict runs."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Add src to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from src import global_utils as utils

log = utils.get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Instantiate the datamodule and iterate over its dataloaders to presave batches to disk.

    Depending on `cfg.project`, either presaves the train/val/test dataloaders, or the
    predict dataloader. Iterating triggers each dataset's `__getitem__` and `collate_fn`,
    which are responsible for writing the presaved data to disk.

    Args:
        cfg (DictConfig): Hydra configuration composed from the config group.
    """

    utils.print_config(cfg, resolve=True)

    cfg.datamodule.instance.mode = "save"  # Force mode to 'save' for presaving
    cfg.datamodule.instance.persistent_workers = (
        False  # Disable persistent workers for saving as we just iterate once
    )
    cfg.datamodule.instance.batch_size = cfg.datamodule.dataset.batch_size_save

    save_dir = Path(cfg.datamodule.dataset.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not "pred" in cfg.project:
        log.info("Train/Val/Test presave stage")
        cfg.datamodule.dataset.predict_dataset = None

        log.info(f"Instantiating Datamodule in mode: {cfg.datamodule.instance.mode}")
        datamodule = hydra.utils.instantiate(cfg.datamodule.instance)

        log.info("Preparing data...")
        datamodule.prepare_data()

        log.info("Setting up data (fit stage)...")
        datamodule.setup(stage="fit")

        if datamodule.train_dataloader():
            log.info(f"Processing Train Dataset with {datamodule.hparams.num_workers} workers...")
            # We iterate to trigger __getitem__ (save) and collate_fn (save to disk)
            for _ in tqdm(
                datamodule.train_dataloader(),
                desc="Saving Train Batches",
                total=len(datamodule.train_dataloader()),
            ):
                pass

        if datamodule.val_dataloader():
            log.info(f"Processing Val Dataset with {datamodule.hparams.num_workers} workers...")
            for _ in tqdm(
                datamodule.val_dataloader(),
                desc="Saving Val Batches",
                total=len(datamodule.val_dataloader()),
            ):
                pass

        log.info("Setting up data (test stage)...")
        datamodule.setup(stage="test")

        if datamodule.test_dataloader():
            log.info(f"Processing Test Dataset with {datamodule.hparams.num_workers} workers...")
            for _ in tqdm(
                datamodule.test_dataloader(),
                desc="Saving Test Batches",
                total=len(datamodule.test_dataloader()),
            ):
                pass
    else:
        log.info("Predict presave stage")
        cfg.datamodule.dataset.train_dataset = None
        cfg.datamodule.dataset.val_dataset = None
        cfg.datamodule.dataset.test_dataset = None

        log.info(f"Instantiating Datamodule in mode: {cfg.datamodule.instance.mode}")
        datamodule = hydra.utils.instantiate(cfg.datamodule.instance)

        log.info("Preparing data...")
        datamodule.prepare_data()

        log.info("Setting up data (predict stage)...")
        datamodule.setup(stage="predict")

        if datamodule.predict_dataloader():
            log.info(
                f"Processing Predict Dataset with {datamodule.hparams.num_workers} workers..."
            )
            for _ in tqdm(
                datamodule.predict_dataloader(),
                desc="Saving Predict Batches",
                total=len(datamodule.predict_dataloader()),
            ):
                pass

    log.info("Presave completed successfully!")


if __name__ == "__main__":
    main()
