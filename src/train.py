"""Module contains classes and functions that helps with training and evalutaing models."""
import tempfile
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def train_epoch(model, dataloader, loss_fn, optimizer, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    epoch_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        if isinstance(inputs, list):
            inputs = (i.to(device) for i in inputs)
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Received prediction of shape {predictions.shape} and targets of shape {targets.shape}")
        loss = loss_fn(predictions, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, mean=None, std=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if isinstance(inputs, list):
                inputs = (i.to(device) for i in inputs)
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Received prediction of shape {predictions.shape} and targets of shape {targets.shape}")
            if (mean and std) is not None:
                predictions = predictions * std + mean
                targets = targets * std + mean
            batch_size = targets.shape[0]
            test_loss += batch_size * loss_fn(predictions, targets)
    return test_loss.item() / len(dataloader.dataset)


def evaluate_per_timestep(model, dataloader, loss_fn, horizon, mean=None, std=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = torch.zeros(horizon).to(device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            if isinstance(inputs, list):
                inputs = (i.to(device) for i in inputs)
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Received prediction of shape {predictions.shape} and targets of shape {targets.shape}")
            if (mean and std) is not None:
                predictions = predictions * std + mean
                targets = targets * std + mean
            batch_size = targets.shape[0]
            test_loss += batch_size * torch.mean(loss_fn(predictions, targets), dim=0).squeeze()
    return test_loss / len(dataloader.dataset)


def predict(model, dataloader, device=None, return_full=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    pred_list, target_list, input_list = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets = targets.to(device)
            if isinstance(inputs, list):
                input_list.append(inputs[0])
            else:
                input_list.append(inputs)
            target_list.append(targets)
            pred_list.append(model(inputs))
    if return_full:
        return torch.cat(pred_list), torch.cat(input_list), torch.cat(target_list)
    return torch.cat(pred_list)


def plot_sample(model, dataset, mean, std, device='cpu', fig_name=None):
    model.to(device)
    random_indices = (torch.randint(0, len(dataset), size=(1 * 1, )))
    subset = torch.utils.data.Subset(dataset, random_indices)
    loader = DataLoader(subset)
    preds, inputs, targets = predict(model, loader, device=device, return_full=True)
    preds = preds.to('cpu') * std + mean
    inputs = inputs.to('cpu') * std + mean
    targets = targets.to('cpu') * std + mean
    # TODO: Extracting previous values based on the assumption that the first column
    # is the target column. Not the most robust solution.
    inputs = inputs[:, :, 0].unsqueeze(dim=2)
    fig, ax = plt.subplots(1, 1, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.set_ylabel('Reglulating Power [MWh]')
    ax.set_xlabel('Hours')
    sns.lineplot(x=torch.arange(-inputs.shape[1], 0), y=inputs.squeeze(), ax=ax, marker='X', linewidth=4)
    sns.lineplot(x=torch.arange(0, preds.shape[1]), y=preds.squeeze(),
                 ax=ax, marker='X', label='Predictions', linewidth=4)
    sns.lineplot(x=torch.arange(0, targets.shape[1]), y=targets.squeeze(),
                 ax=ax, marker='X', label='Labels', linewidth=4)
    matplotlib.rcParams.update({'font.size': 22})
    plt.savefig(f'{fig_name}.pdf')


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_loss = []
        self.val_loss = []

    def train_print(self, x, verbose):
        if verbose:
            print(x)

    def fit(self, epochs, train_loader, loss_fn, optimizer, val_loader=None, early_stopping_patience=10, device=None, verbose=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        early_stopper = EarlyStopper(self.model, early_stopping_patience, restore=True)
        for e in range(epochs):
            self.train_print(f"Epoch {e + 1}:", verbose)
            train_loss = train_epoch(self.model, train_loader, loss_fn, optimizer, device=device)
            self.train_loss.append(train_loss)
            self.train_print(f"Training loss: {train_loss}", verbose)
            if val_loader:
                val_loss = evaluate(self.model, val_loader, loss_fn, device=device)
                self.val_loss.append(val_loss)
                self.train_print(f"Validation loss: {val_loss}", verbose)
                if early_stopper.should_stop(e, val_loss):
                    self.train_print(f"Stopped by early stopper after {e + 1} epochs.", verbose)
                    if early_stopper.restore:
                        cp = torch.load(early_stopper.checkpoint.name)
                        self.train_print(
                            f"Loading parameters from epoch {cp['epoch'] + 1} with validation loss {cp['val_loss']}", verbose)
                        self.model.load_state_dict(cp['model_state_dict'])
                        early_stopper.checkpoint.close()
                        break
            self.train_print('', verbose=verbose)

    def plot_training_losses(self, path=None, name=None):
        if self.train_loss:
            a = sns.lineplot(self.train_loss, label='Training loss')
            a.set_xlabel('Epoch')
            a.set_ylabel('Loss')
        if self.val_loss:
            a = sns.lineplot(self.val_loss, label='Validation loss')
        if path:
            plt.savefig(f'{path}/{name}_training_loss.pdf')
        else:
            plt.show()


class EarlyStopper:
    def __init__(self, model, patience, restore=True) -> None:
        self.model = model
        self.patience = patience
        self.restore = restore
        if restore:
            self.checkpoint = tempfile.NamedTemporaryFile()
        self.counter = 0
        self.min_val_loss = float('inf')

    def should_stop(self, epoch, val_loss):
        if val_loss < self.min_val_loss:
            self.counter = 0
            self.min_val_loss = val_loss
            if self.restore:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, self.checkpoint.name)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
