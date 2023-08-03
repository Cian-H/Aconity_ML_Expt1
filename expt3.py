# From expt2 selected trials ???
# Data handling imports
from dask.distributed import Client, LocalCluster
import dask.array as da

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from expt1 import (
    Model,
    device,
    X_train,
    y_train,
    X_val,
    y_val,
    collate_fn,
)
from custom_activations import SoftExp, PBessel

# Suppress some warning messages from pytorch_lightning,
# It really doesn't like that i've forced it to handle a dask array!
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning, module=pl.__name__)
# Also, set up a log to record debug messages for failed trials
logging.basicConfig(filename="debug.log", encoding="utf-8", level=logging.ERROR)

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)


    # Prepare datasets
    train = DataLoader(
        list(zip(X_train.values(), y_train.values())),
        collate_fn=collate_fn,
        shuffle=True,
    )
    valid = DataLoader(
        list(zip(X_val.values(), y_val.values())),
        shuffle=True,
        collate_fn=collate_fn,
    )
    
# Set up the model architecture and other necessary components
model = Model(
    # Training parameters
    optimizer=optim.Adam,
    # Model parameters
    compressor_kernel_size=128,
    compressor_chunk_size=128,
    compressor_act=(SoftExp, (), {}),
    conv_kernel_size=128,
    conv_act=(nn.Tanh, (), {}),
    conv_norm=False,
    channel_combine_act=(nn.Softplus, (), {}),
    param_ff_depth=2,
    param_ff_width=16,
    param_ff_act=(PBessel, (), {}),
    ff_width=1024,
    ff_depth=6,
    ff_act=(nn.Softplus, (), {}),
    out_size=2,
    out_act=(nn.Sigmoid, tuple(), dict()),
).to(device)

if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=15, verbose=False, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=10,
        mode="min",
    )

    logger = WandbLogger(project="Aconity_ML_Expt1", name="Test 3")
    logger.experiment.watch(model, log="all", log_freq=1)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=-1,
        devices="auto",
        strategy="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,  # Disabled or we get error because X is dask array
    )
    # Finally, train the model
    trainer.fit(model, train, valid)
