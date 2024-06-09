import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import Linear, GATModel, EncoderDecoder
import utils
from train import Trainer, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='forecast trainer', description='Train and evaluate a model.',)
    # Required arguments:
    parser.add_argument('--lookback_window', type=int, required=True, help='Size of lookback window.')
    parser.add_argument('--horizon', type=int, required=True, help='Size of horizon.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['linear', 'gat_model', 'encoder_decoder'], help='Model type.')

    # Optional arguments:
    parser.add_argument('--data_path', type=str, required=False, help='Path to data file.')
    parser.add_argument('--target_column', required=False,
                        choices=['NO1_mFRR', 'NO2_mFRR', 'NO3_mFRR', 'NO4_mFRR', 'system_imbalance'], default='system_imbalance')
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--patience', type=int, required=False, default=20, help='Early stopping patience.')
    parser.add_argument('--silent', action='store_false')
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
    TRAIN_MEAN = train.mean()
    TRAIN_STD = train.std()
    train, val, test = utils.standardize([train, val, test], TRAIN_MEAN, TRAIN_STD)

    future_cols = None
    if args.model == 'encoder_decoder':
        if country == 'Norway':
            future_cols = list(filter(lambda x: 'forecasted' in x or 'scheduled' in x, train.columns))
        elif country == 'Belgium':
            future_cols = list(filter(lambda x: 'fc' in x, train.columns))

    # Create Dataset and Dataloaders
    train_ds = utils.TimeSeriesDataset(inputs=train.drop(columns=future_cols) if future_cols else train,
                                       future_inputs=train[future_cols] if future_cols else None,
                                       targets=train[[args.target_column]],
                                       lookback_window=args.lookback_window,
                                       horizon=args.horizon)

    val_ds = utils.TimeSeriesDataset(inputs=val.drop(columns=future_cols) if future_cols else val,
                                     future_inputs=val[future_cols] if future_cols else None,
                                     targets=val[[args.target_column]],
                                     lookback_window=args.lookback_window,
                                     horizon=args.horizon)

    test_ds = utils.TimeSeriesDataset(inputs=test.drop(columns=future_cols) if future_cols else test,
                                      future_inputs=test[future_cols] if future_cols else None,
                                      targets=test[[args.target_column]],
                                      lookback_window=args.lookback_window,
                                      horizon=args.horizon)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    num_features = len(train.columns)
    if args.model == 'gat_model':
        model = GATModel(input_steps=args.lookback_window, num_features=num_features, output_steps=args.horizon)
    elif args.model == 'linear':
        model = Linear(args.lookback_window * num_features, args.horizon)
    elif args.model == 'encoder_decoder':
        model = EncoderDecoder(len(train.columns) - len(future_cols), len(future_cols),
                               128, out_features=1, enc_dropout=0.2, dec_dropout=0)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model)
    loss = nn.MSELoss()
    trainer.fit(args.epochs, train_loader, loss, optim, val_loader,
                early_stopping_patience=args.patience, verbose=args.silent)

    # Evaluate model:
    print(evaluate(model, test_loader, nn.MSELoss(), TRAIN_MEAN[args.target_column], TRAIN_STD[args.target_column]))
