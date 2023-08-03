# %% [markdown]
# <h1>Experiment 1</h1>
# <h3>Initial hyperparameter tuning</h3>
# <p>Summary</p>
# <ul>
# <li>A model was created with a dynamic constructor, allowing for a hyperparameter-driven model</li>
# <li>Hyperparameters were tuned using <code>`Optuna`</code></li>
# <li>Training loop was constructed using <code>`PyTorchLightning`</code></li>
# <li>Model was trained on a cluster of machines using a shared SQL trial database</li>
# <li>An extremely aggressive pruning algorithm was used to quickly narrow in on an optimal hyperparameter space</li>
# <li>Experiment 1 was left to train on the cluster for 2 days</li>
# </ul>

# %%
# Data handling imports
from dask.distributed import Client, LocalCluster
import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np
import pickle
import random
from itertools import chain
from tqdm.auto import tqdm

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from pytorch_lightning import Trainer
import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback


# Suppress some warning messages from pytorch_lightning,
# It really doesn't like that i've forced it to handle a dask array!
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=pl.__name__)

# Also, set up a log to record debug messages for failed trials
import logging

logging.basicConfig(filename="debug.log", encoding="utf-8", level=logging.ERROR)

# %% [markdown]
# <h3>Patching PyTorchLightning</h3>
# <p>
# A key part of this project was to develop a patch for PyTorchLightning to allow for the use of <code>`dask`</code> arrays as inputs. It was important that PyTorchLightning can accept <code>`dask`</code> arrays and only load the data into memory when needed. Otherwise, our extremely large datasets would simply crash our system as they are significantly larger than the available RAM and VRAM.
# </p><p>
# After several versions of the patch, this final version was developed. It is a simple monkey patch that wraps the <code>pytorch_lightning.utlities.data._extract_batch_size</code> generator with a check that mimics the expected behaviour for torch tensors when given a dask array and extends its type signature to ensure static analysis is still possible.
# </p><p>
# With this patch applied, the forward method in our model can accept a dask array and only compute each chunk of the array when needed. This allows us to train our model on datasets that are significantly larger than the available memory.
# </p>

# %%
# Monkey patch to allow pytorch lightning to accept a dask array as a model input
from typing import Any, Generator, Iterable, Mapping, Optional, Union

BType = Union[da.Array, torch.Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

unpatched = pl.utilities.data._extract_batch_size


def patch(batch: BType) -> Generator[Optional[int], None, None]:
    if isinstance(batch, da.core.Array):
        if len(batch.shape) == 0:
            yield 1
        else:
            yield batch.shape[0]
    else:
        yield from unpatched(batch)


pl.utilities.data._extract_batch_size = patch

# %%
# Set the device to use with torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Prepare a dask cluster and client
def create_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    return client

if __name__ == "__main__":
    client = create_client()

# %%
# Load X and y for training
samples = list(range(1, 82))

with open("sample_X.pkl", "rb") as f:
    X = pickle.load(f)

with open("sample_y.pkl", "rb") as f:
    y = pickle.load(f)

# %% [markdown]
# <h3>Dataset Splitting</h3>
# <p>The dataset is split into a training and validation dataset (80:20 split). Because the number of available samples is extremely small, we haven't produced a test dataset. In the future, as more data is obtained, a test set should be included whenever possible.</p>

# %%
# Separate samples into training and validation sets
val_samples = random.sample(samples, k=len(samples) // 5)
train_samples = [s for s in samples if s not in val_samples]

X_train = {i: X[i] for i in train_samples}
X_val = {i: X[i] for i in val_samples}
y_train = {i: y[i] for i in train_samples}
y_val = {i: y[i] for i in val_samples}

# %% [markdown]
# <h3>Dataset Collation</h3>
# <p>This function returns a closure for collating our data in a torch DataLoader. The use of a DataLoader will allow us to shuffle and prefetch data, reducing overfitting and maximising performance as IO will be a bottleneck. The closure is dynamically constructed, allowing us to select the outputs we train against. However, for this experiment we will match against all outputs for simplicity.</p>

# %%
# Create a function to dynamically modify data collation
def collate_fn(batch):
    X0 = batch[0][0][0].to_numpy(dtype=np.float32)[0]
    X1 = batch[0][0][1].to_dask_array(lengths=True)
    y = batch[0][1].to_numpy(dtype=np.float32)
    return (
        torch.from_numpy(X0).to(device),
        X1,
        torch.from_numpy(y).to(device),
    )

# %% [markdown]
# <h3>Convolutional Data Compression</h3>
# <p>
# The <code>`DaskCompression`</code> module accepts a dask array, and applies a convolutional kernel to it to significantly compress the input data. This allows us to transform a larger than VRAM dataset into one that can fit on our GPU, and (hopefully) retain the relevant information to train the rest of our model on.
# </p><p>
# Note how the kernel is only computed in line 12 and is immediately compressed via convolution. This ensures that only one kernel needs to be stored in memory at a time, avoiding the need to hold the entire dataset in memory at once.
# </p>

# %%
class DaskCompression(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, chunk_size=1, device=device
    ):
        super(DaskCompression, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.device = device
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size).to(device)

    def compress_kernel(self, kernel):
        return (
            self.conv(torch.from_numpy(kernel.compute()).to(self.device))
            .squeeze()
            .to("cpu")  # return to cpu to save VRAM
        )

    def forward(self, x):
        # Precompute the dimensions of the output array
        dim0, dim2 = x.shape
        assert dim2 == self.in_channels
        dim0 = (dim0 // self.kernel_size) // self.chunk_size
        x = x.reshape(dim0, self.chunk_size, self.kernel_size, dim2)
        x = da.transpose(x, axes=(0, 1, 3, 2))

        x = [self.compress_kernel(kernel) for kernel in x]
        return torch.stack(x).to(self.device)



# %% [markdown]
# <h3>Model Design</h3>
# <p>
# The model was designed to be a dynamically constructed, hyperparameter driven model for ease of hyperparameter optimisation. The contructed model will process data in the following way:
# </p>
# <ol>
# <li>The input is left/right padded to a multiple of the compressor kernel size</li>
# <li>The dask array is compressed by a <code>`DaskCompressor`</code> layer, treating each input as a channel</li>
# <li>The compressed array is then recursively convoluted down to a size less than or equal to the width of our feedforward network</li>
# <li>The channels of the now convolved data are combined</li>
# <li>The combined, flattened data is then left/right padded to the width of the feedforward network</li>
# <li>Finally, the data is fed into a feedforward network</li>
# </ol>
# <p>
# This relatively simple design allows the network to accept both larger-than-ram datasets as inputs, and datasets of variable sizes. This makes it suitable for training on whole Aconity datasets, without the need for culling or binning.
# </p>

# %%
class Model(pl.LightningModule):
    def __init__(
        self,
        # pl attributes
        optimizer=torch.optim.Adam,
        optimizer_args=(),
        optimizer_kwargs={},
        scheduler=None,
        scheduler_kwargs={},
        loss=torch.nn.MSELoss(),
        train_ds=None,
        val_ds=None,
        # model args & kwargs
        compressor_kernel_size=128,
        compressor_chunk_size=128,
        compressor_act=(nn.ReLU, (), {}),
        conv_kernel_size=128,
        conv_norm=False,
        conv_act=(nn.ReLU, (), {}),
        channel_combine_act=(nn.ReLU, (), {}),
        param_ff_depth=4,
        param_ff_width=16,
        param_ff_act=(nn.ReLU, (), {}),
        ff_width=512,
        ff_depth=4,
        ff_act=(nn.ReLU, (), {}),
        out_size=6,
        out_act=(nn.ReLU, (), {}),
    ):
        super().__init__()
        # Assign necessary attributes for pl model
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.loss = loss
        self.mse = MeanSquaredError()
        self.train_ds = train_ds
        self.val_ds = val_ds
        # Attrs for dynamically created model to be tested
        self.compressor_kernel_size = compressor_kernel_size
        self.compressor_chunk_size = compressor_chunk_size
        self.conv_kernel_size = conv_kernel_size
        self.ff_width = ff_width
        self.ff_depth = ff_depth
        self.out_size = out_size
        # layers
        # compressor compresses and converts dask array to torch tensor
        self.convolutional_compressor = DaskCompression(
            5,
            5,
            kernel_size=compressor_kernel_size,
            chunk_size=compressor_chunk_size,
        )
        self.compressor_act = compressor_act[0](*compressor_act[1], **compressor_act[2])
        # convolutional layer recursively applies convolutions to the compressed input
        self.conv = nn.Conv1d(5, 5, kernel_size=conv_kernel_size)
        self.conv_norm = nn.LocalResponseNorm(5) if conv_norm else nn.Identity()
        self.conv_act = conv_act[0](*conv_act[1], **conv_act[2])
        self.combine_channels = nn.Conv1d(5, 1, kernel_size=1)
        self.channel_combine_act = channel_combine_act[0](
            *channel_combine_act[1], **channel_combine_act[2]
        )
        self.param_ff = nn.Sequential(
            nn.Linear(4, param_ff_width),
            param_ff_act[0](*param_ff_act[1], **param_ff_act[2]),
            *chain(
                *(
                    (
                        nn.Linear(param_ff_width, param_ff_width),
                        param_ff_act[0](*param_ff_act[1], **param_ff_act[2]),
                    )
                    for _ in range(param_ff_depth)
                )
            ),
        )
        self.ff = nn.Sequential(
            nn.Linear(ff_width + param_ff_width, ff_width),
            ff_act[0](*ff_act[1], **ff_act[2]),
            *chain(
                *(
                    (
                        nn.Linear(ff_width, ff_width),
                        ff_act[0](*ff_act[1], **ff_act[2]),
                    )
                    for _ in range(ff_depth)
                )
            ),
        )
        self.out_dense = nn.Linear(ff_width, out_size)
        self.out_act = out_act[0](*out_act[1], **out_act[2])

    @staticmethod
    def pad_ax0_to_multiple_of(x, multiple_of):
        padding = (((x.shape[0] // multiple_of) + 1) * multiple_of) - x.shape[0]
        left_pad = padding // 2
        right_pad = padding - left_pad
        return da.pad(
            x, ((left_pad, right_pad), (0, 0)), mode="constant", constant_values=0
        )

    def pad_to_ff_width(self, x):
        padding = self.ff_width - x.shape[1]
        left_pad = padding // 2
        right_pad = padding - left_pad
        return F.pad(
            x,
            (right_pad, left_pad, 0, 0),
            mode="constant",
            value=0.0,
        )

    def forward(self, x0, x1):
        # pad to a multiple of kernel_size * chunk_size
        x1 = self.pad_ax0_to_multiple_of(
            x1, self.compressor_kernel_size * self.compressor_chunk_size
        )
        x1 = self.convolutional_compressor(x1)
        x1 = x1.reshape(x1.shape[0] * x1.shape[1], x1.shape[2]).T.unsqueeze(0)
        while x1.shape[2] > self.ff_width:
            x1 = self.conv(x1)
            x1 = self.conv_norm(x1)
            x1 = self.conv_act(x1)
        x1 = self.combine_channels(x1)
        x1 = self.channel_combine_act(x1)
        x1 = x1.squeeze(1)
        x1 = self.pad_to_ff_width(x1)
        x0 = x0.unsqueeze(0)
        x0 = self.param_ff(x0)
        x = torch.cat((x1, x0), dim=1)
        x = self.ff(x)
        x = self.out_dense(x)
        x = self.out_act(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), *self.optimizer_args, **self.optimizer_kwargs
        )
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            return optimizer, scheduler
        else:
            return optimizer

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        y_hat = self(x0, x1)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        mse = self.mse(y_hat, y)
        self.log('train_MSE', mse, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        y_hat = self(x0, x1)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        mse = self.mse(y_hat, y)
        self.log('train_MSE', mse, on_step=True, on_epoch=True, prog_bar=True)

# %% [markdown]
# <h3>Activation Functions</h3>
# <p>
# For our hyperparameter optimisation, we intend to test all the activation functions in PyTorch. In addition to the builtin activations, we will also train using the following custom implemented activation functions from literature or our own design:
# </p>
# <ol>
# <li><b><code>BMU</code>:</b> Bio-Mimicking Unit; an activation function designed to mimic the activation potential of a biological neuron.</li>
# <li><b><code>SoftExp</code>:</b> Soft Exponential function; a parametric activation function that fits to a wide variety of exponential curves (DOI: <a href=https://arxiv.org/abs/1602.01321v1>10.48550/arXiv.1602.01321</a>)</li>
# <li><b><code>LeakyPReQU</code>:</b> Leaky Parametric Rectified Quadratic Unit; A smoothly and continuously differentiable function that is a parametrically sloped line for <code>x&#8924;0</code> and a quadratic curve for <code>x&gt;0</code></li>
# <li><b><code>ISRU</code>:</b> Inverse Square Root Unit; a somewhat uncommon function that can be useful in models such as this as it yields a continuously differentiable curve while being extremely fast to compute using bit manipulation</li>
# <li><b><code>ISRLU</code>:</b> Inverse Square Root Linear Unit; a modified ISRU that is an ISRU for <code>x&lt;0</code> and <code>`f(x)=x`</code> for <code>x&#8925;0</code> (DOI: <a href=https://arxiv.org/abs/1710.09967>10.48550/arXiv.1710.09967</a>)</li>
# <li><b><code>PBessel</code>:</b> Parametric Besse; A parametric Bessel curve yielding various different wave formations depending on a trainable parameter</li>
# <li><b><code>Sinusoid</code>:</b> A parametric sine wave, with amplitude and wavelength as trainable parameters</li>
# <li><b><code>Modulo</code>:</b> A parametric sawtooth wave, <code>`f(x)=x%&#593;</code> where &#593; is a trainable parameter</li>
# <li><b><code>TriWave</code>:</b> A parametric triangle wave, with amplitude and wavelength as trainable parameters</li>
# <li><b><code>Gaussian</code>:</b> A parametric gaussian curve, with trainable amplitude</li>
# </ol>

# %%
# Create a dispatcher including all builtin activations and
# Several custom activations from experimentation or literature
from custom_activations import SoftExp, PBessel


activation_dispatcher = {
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "Softplus": nn.Softplus,
    "SoftExp": SoftExp,
    "PBessel": PBessel,
}

# %% [markdown]
# <h3>Hyperparameter training</h3>
# <p>Here, we define an objective function, describing what we want Optuna to do during each trial and how to react to various errors and/or situations that may arise. To summarise the objective:</p>
# <ul>
# <li>Optuna selects hyperparameters for all input parameters within the given constraints</li>
# <li>A model is generated using the selected hyperparameters</li>
# <li>PyTorchLightning trains the model through 2 epochs</li>
# <li>The model is evaluated on the validation set</li>
# <li>The validation loss is returned to Optuna</li>
# </ul>
# <p>
# Optuna monitors the reported validation loss and attempts to minimise it. An extremely aggressive pruning strategy known as "hyperband pruning" is used to efficiently reduce down the parameter space to something more reasonable. Any parameter set which optuna deems suboptimal will be immediately pruned or even stopped early to save time.
# </p>

# %%
# Test parameters
n_epochs = 2
output_keys = list(next(iter(y_train.values())).keys())
activation_vals = list(activation_dispatcher.keys())


# Next we define the objective function for the hyperparameter optimization
def objective(trial):
    torch.cuda.empty_cache()
    objective_value = torch.inf
    model = None
    logger = None
    try:
        # Select hyperparameters for testing
        compressor_kernel_size = 128
        compressor_chunk_size = 128
        compressor_act = (
            activation_dispatcher[
                trial.suggest_categorical("compressor_act", activation_vals)
            ],
            (),
            {},
        )
        conv_kernel_size = 128
        conv_norm = trial.suggest_categorical("conv_norm", [True, False])
        conv_act = (
            activation_dispatcher[
                trial.suggest_categorical("conv_act", activation_vals)
            ],
            (),
            {},
        )
        channel_combine_act = (
            activation_dispatcher[
                trial.suggest_categorical("channel_combine_act", activation_vals)
            ],
            (),
            {},
        )
        param_ff_depth = trial.suggest_int("param_ff_depth", 2, 8, 2)
        param_ff_width = trial.suggest_int("param_ff_width", 16, 64, 16)
        param_ff_act = (
            activation_dispatcher[
                trial.suggest_categorical("param_ff_act", activation_vals)
            ],
            (),
            {},
        )
        ff_width = trial.suggest_int("ff_width", 256, 1025, 256)
        ff_depth = trial.suggest_int("ff_depth", 2, 8, 2)
        ff_act = (
            activation_dispatcher[trial.suggest_categorical("ff_act", activation_vals)],
            (),
            {},
        )
        out_size = 2
        out_act = (nn.Sigmoid, tuple(), dict())

        # Set up the model architecture and other necessary components
        model = Model(
            compressor_kernel_size=compressor_kernel_size,
            compressor_chunk_size=compressor_chunk_size,
            compressor_act=compressor_act,
            conv_kernel_size=conv_kernel_size,
            conv_act=conv_act,
            conv_norm=conv_norm,
            channel_combine_act=channel_combine_act,
            param_ff_depth=param_ff_depth,
            param_ff_width=param_ff_width,
            param_ff_act=param_ff_act,
            ff_width=ff_width,
            ff_depth=ff_depth,
            ff_act=ff_act,
            out_size=out_size,
            out_act=out_act,
        ).to(device)

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=n_epochs,
            devices=1,
            logger=logger,
            num_sanity_val_steps=0,  # Needs to be disabled or else we get an error because X is dask array
            # precision="16-mixed",
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
        )
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
        # Finally, train the model
        trainer.fit(model, train, valid)
    except Exception as e:
        logging.exception(f"An exception occurred in trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        if logger is not None:
            logger.experiment.unwatch(model)
            logger.experiment.finish()
    del model
    torch.cuda.empty_cache()
    if objective_value == torch.inf:
        raise optuna.exceptions.TrialPruned()
    return objective_value

# %% [markdown]
# <h3>Hyperparameter Optimisation on a Computing Cluster</h3>
# <p>
# The final important step is to run the optimisation using a cluster of computers to maximise the number of trials that can be run in parallel. Although this could be achieved using a more complex, scheduler controlled system and dask, we will use the far simpler approach of using a shared SQL ledger to keep track of the trials and their results. This is a very simple approach, but it is sufficient for our purposes, and is easy to implement. Using this approach, the model was trained on a cluster of 5 computers at once.
# </p>

# %%
if __name__ == "__main__":
    # storage_name = "sqlite:///optuna.sql"
    storage_name = "mysql+pymysql://root:Ch31121992@192.168.1.10:3306/optuna_db"
    study_name = "Composition Experiment 1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=None,
        timeout=None,
    )


