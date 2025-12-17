import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import seed_everything
from src import global_utils as utils

log = utils.get_logger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(config: DictConfig) :

    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    OmegaConf.set_struct(config, True)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    # TO avoid useless initialization of the datasets
    config.datamodule.dataset.predict_dataset = None
    datamodule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning module
    log.info(f"Instantiating module <{config.module.instance._target_}>")
    module = hydra.utils.instantiate(config.module.instance)

    # Init lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger)
        
    # Train the model
    if config.get("ckpt_path") is not None or config.get("ckpt_path") != "last":
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)

            if "state_dict" in checkpoint:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint, strict=False)
            
            log.info(f"Missing keys in checkpoint: {missing_keys}")
            log.info(f"Unexpected keys in checkpoint: {unexpected_keys}")

            ckpt_path = None
        
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    
    elif ckpt_path == "last" :
        log.info(f"Starting training from last checkpoint {ckpt_path} !")
    
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None
    
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
   
    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    
    return None

if __name__ == "__main__":
    train()