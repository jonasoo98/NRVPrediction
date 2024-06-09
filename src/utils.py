"""Module contains utility functions and classes."""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def add_cyclic_representation(df, features):
    """
    Add sine and cosine representation of features.

    Args:
        df (pd.DataFrame): Dataframe to add representations to.
        features (list[str]): List of features to add cyclyc representation for.
    """
    df = df.copy()
    for f in features:
        attr = getattr(df.index, f)
        df[f'sin_{f}'] = np.sin(attr * 2 * np.pi / attr.nunique())
        df[f'cos_{f}'] = np.cos(attr * 2 * np.pi / attr.nunique())
    return df


def belg_get_train_val_test_split(df):
    """
    Split a dataframe with data from the Belgian market into three seperate
    dataframes.

    Args:
        df (pd.DataFrame): Dataframe to Split
    """
    train = df[(df.index >= '2014-01-01') & (df.index < '2017-01-01')].copy()
    val = df[(df.index >= '2017-01-01') & (df.index < '2018-01-01')].copy()
    test = df[(df.index >= '2018-01-01') & (df.index < '2018-03-01')].copy()
    return train, val, test


def nor_get_train_val_test_split(df):
    """
    Split dataframe with data from the Norwegian market into three sepearte dataframes.

    Args:
        df (pd.DataFrame): Dataframe to split.
    """
    train = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')].copy()
    val = df[(df.index >= '2023-01-01') & (df.index < '2023-06-01')].copy()
    test = df[(df.index >= '2023-06-01') & (df.index < '2023-09-01')]
    return train, val, test


def standardize(df_list, mean, std):
    """
    Apply z-score standardization to a list of dataframes.

    Args:
        df_list (list[pd.DataFrame]): List of dataframes.
        mean (pd.Series): Series with the mean of all values in the dataframes.
        std (pd.Series): Series with the standard deviation of all values in the dataframes.

    Returns:
        list[pd.DataFrame]: List of standardized dataframes.
    """
    for i, df in enumerate(df_list):
        df_list[i] = (df - mean) / std
    return df_list


def to_tensor(x):
    """
    Convert input to tensor.

    Args:
        x (pd.DataFrame | np.ndarray): Input to convert.

    Returns:
        torch.Tensor: Input converted to tensor.
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values
    if x.ndim < 2:
        x = x.reshape(x.shape[0], 1)  # Add dim so that shape is always (timesteps, features).
    return torch.Tensor(x)


class TimeSeriesDataset(Dataset):
    """
    Dataset class that creates a sliding window over the provided data.
    """

    def __init__(self, inputs, targets, future_inputs=None, lookback_window=1, horizon=1, shift=0):
        """
        Args:
            inputs (pd.Dataframe | np.ndarray): Model input.
            targets (pd.Dataframe | np.ndarray): Model targets.
            future_inputs (pd.Dataframe | np.ndarray): Future inputs. This is typically used for an encoder-decoder
                model where you want to feed the encoder and decoder with different input (default None).
            lookback_window (int): size of lookback window (default 1).
            horizon (int): Size of horizone (default 1)
            shift (int): Number of timesteps between the last step in the
                lookback_window and the first timestep in the horizon (default 0).
        """
        super().__init__()
        self.inputs, self.targets = to_tensor(inputs), to_tensor(targets)
        if future_inputs is not None:
            future_inputs = to_tensor(future_inputs)
        self.future_inputs = future_inputs
        self.lookback_window = lookback_window
        self.horizon = horizon
        self.shift = shift
        self.total_window_size = lookback_window + shift + horizon

    def __len__(self):
        return len(self.inputs) - self.total_window_size + 1

    def __getitem__(self, index):
        inputs = self.inputs[index: index + self.lookback_window]
        targets = self.targets[index + self.lookback_window + self.shift: index + self.total_window_size]
        if self.future_inputs is not None:
            future_inputs = self.future_inputs[index + self.lookback_window +
                                               self.shift: index + self.total_window_size]
            return (inputs, future_inputs), targets
        return inputs, targets
