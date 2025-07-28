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
# from baseline.baseline import Baseline


def main(config, seed):
    wandb.finish()
    #pl.seed_everything(config['seed'], workers=True)
    pl.seed_everything(seed, workers=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config['experiment']}_{timestamp}"

    ckpt_path = f"{config['ckpt_dir']}/seed-{seed}"

    wandb_logger = WandbLogger(project=config['project'], log_model=config['log_model'], config=config, name=run_name)
    checkpoint_callback = ModelCheckpoint(
        monitor=config['monitor'],
        dirpath=ckpt_path,
        filename=config['ckpt_filename'],
        save_top_k=config['save_top_k'],
        mode=config['ckpt_mode']
    )

    vae = VAE(hyperparams=config)
    data_module = HeatDataModule(hyperparams=config)

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        strategy=config['strategy'],
        devices=config['devices'],
        precision=config['precision'],
        max_epochs=config['max_epochs'],
        profiler=config['profiler'],
        logger=None, #wandb_logger,
        log_every_n_steps=config['log_every_n_steps'],
        callbacks=[checkpoint_callback]
    )

    trainer.fit(vae, data_module)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config')
    parser.add_argument('--seed', type=int, required=True, help='seed value')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if not os.path.exists(f"{config['data_dir']}/train_data.npy" and f"{config['data_dir']}/val_data.npy"):
        print("Generating synthetic data...")
        generate_synthetic_data(config)

    main(config, args.seed)
