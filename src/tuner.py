"""Module for tuning hyperparameteres."""
import os
import tempfile
from functools import partial
import argparse
from torch import nn
from torch.utils.data import DataLoader
import torch
from ray.train import Checkpoint
from ray import tune
from ray import train as ray_train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import pandas as pd
import utils
from models import GATModel
from train import train_epoch, evaluate


def tune_trainer(config, data_loaders, train_mean, train_std):
    train_loader, test_loader = data_loaders
    loss = nn.MSELoss()
    num_features = train_loader.dataset[0][0].shape[1]
    lookback_window = train_loader.dataset[0][0].shape[0]
    horizon = train_loader.dataset[0][1].shape[0]
    model = GATModel(input_steps=lookback_window,
                     num_features=num_features,
                     output_steps=horizon,
                     gru_n_layers=config['gru_n_layers'],
                     gru_hid_dim=config['gru_hid_dim'],
                     forecast_hid_dim=config['forecast_hidden_dim'],
                     dropout=config['dropout'])

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for i in range(80):  # Each trial will run for maxium 80 epochs.
        train_epoch(model, train_loader, loss, optim)
        test_loss = evaluate(model, test_loader, loss, train_mean, train_std)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                # Create a cp every 5 epochs.
                # This saves the model to the trial directory
                torch.save(
                    {
                        'epoch': i + 1,
                        'model': model.state_dict(),
                        'optimizer': optim.state_dict()
                    },
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            ray_train.report({"test_loss": test_loss}, checkpoint=checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookback_window', type=int, required=True, help='Size of lookback window.')
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--target_column', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    # Read data from file
    data = pd.read_parquet(args.data_path)
    data = data.interpolate('linear', limit=1)  # Fill small gaps with linear interpolation.
    data = data.fillna(0)  # Fill remaining gaps with 0.
    country = 'Norway' if 'NO' in args.target_column else 'Belgium'

    # Add representation for cyclic features:
    granularity = (data.iloc[1].name - data.iloc[0].name).seconds
    if granularity == 3600:  # 1 hour granularity.
        data = utils.add_cyclic_representation(data, ['month', 'day_of_week', 'hour'])
    elif granularity == 900:  # 15 min granularity.
        data = utils.add_cyclic_representation(data, ['month', 'day_of_week', 'quarter'])

    # Split into training, validation and testing:
    if country == 'Norway':
        train, val, test = utils.nor_get_train_val_test_split(data)
    elif country == 'Belgium':
        train, val, test = utils.belg_get_train_val_test_split(data)

    # Standardize:
    train_mean = train.mean()
    train_std = train.std()
    train, val, _ = utils.standardize([train, val, test], train_mean, train_std)

    # Create Dataset and Dataloaders
    train_ds = utils.TimeSeriesDataset(inputs=train,
                                       targets=train[[args.target_column]],
                                       lookback_window=args.lookback_window,
                                       horizon=args.horizon)

    val_ds = utils.TimeSeriesDataset(inputs=val,
                                     targets=val[[args.target_column]],
                                     lookback_window=args.lookback_window,
                                     horizon=args.horizon)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)

    search_space = {
        'gru_n_layers': tune.choice([1, 2, 4, 8]),
        'gru_hid_dim': tune.choice([128, 256]),
        'forecast_hidden_dim': tune.choice([64, 128, 256]),
        'dropout': tune.choice([0, 0.2]),
        'lr': tune.loguniform(1e-4, 1e-1),
    }

    train_func = partial(tune_trainer, data_loaders=(train_loader, val_loader),
                         train_mean=train_mean[args.target_column], train_std=train_std[args.target_column])

    tuner = tune.Tuner(
        train_func,  # Tune with CPU.
        # tune.with_resources(train_func, {"gpu": 0.5}), # Uncomment this line to tune with GPU.
        tune_config=tune.TuneConfig(
            num_samples=30,
            scheduler=ASHAScheduler(metric="test_loss", mode="min", grace_period=8),
            search_alg=HyperOptSearch(
                metric='test_loss',
                mode='min'
            ),
        ),
        param_space=search_space,
    )

    result = tuner.fit()

    best_trial = result.get_best_result(metric="test_loss", mode="min", scope="last")

    print(f"Best trial config: {best_trial.config}")
    print(f"mean_accuracy: {best_trial.metrics['test_loss']}")


if __name__ == '__main__':
    main()
