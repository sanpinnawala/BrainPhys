import argparse
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchinfo import summary
import torch
import numpy as np
import time

from data.data_simulation import generate_synthetic_data
from data.datamodule import HeatDataModule
from model.model import VAE
from baseline.baseline import Baseline


def main(config):
    wandb.finish()
    pl.seed_everything(42, workers=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config['experiment']}_{timestamp}"

    ckpt_path = f"{config['ckpt_dir']}/{config['ckpt_filename']}.ckpt"

    wandb_logger = WandbLogger(project=config['project'], log_model=config['log_model'], config=config, name=run_name)

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        if config['mode'] == 'baseline':
            vae = Baseline.load_from_checkpoint(ckpt_path, hyperparams=config)
        else:
            vae = VAE.load_from_checkpoint(ckpt_path, hyperparams=config)
    else:
        raise ValueError(f"Checkpoint path '{ckpt_path}' does not exist")
    data_module = HeatDataModule(hyperparams=config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=config['log_every_n_steps'],
    )

    trainer.test(vae, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if not os.path.exists(f"{config['data_dir']}/test_data.npy"):
        print("Generating synthetic data...")
        generate_synthetic_data(config)

    main(config)
