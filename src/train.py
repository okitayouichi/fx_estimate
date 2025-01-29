"""
Training process.
"""

import torch
import criterion
import os


def train(data_loader, model, ns_epoch, learn_rates, log_paths, state_paths, device):
    """
    Overall training process including pre-training and fine-tuning.

    Args:
        data_loader(torch.utils.data.DataLoader): Training data loader.
        model(model.FxInversion): Audio effect inversion Model.
        ns_epoch(tuple[int]): List of number of epochs for pre-training and fine-tuning.
        learn_rates(tuple[float]): List of learing rate for pre-training and fine-tuning.
        log_paths(tuple[pathlib.Path]): Path to the log file for pre-training and fine-tuning.
        state_paths(tuple[pathlib.Path]): Path to the model state file for pre-training and fine-tuning.
        device(torch.device): Device on which the tensor calculation is performed.

    Returns:
        model(model.FxInversion): Trained model.
    """
    model.train()
    model = pretrain(data_loader, model, ns_epoch[0], learn_rates[0], criterion.loss_pretrain, log_paths[0], state_paths[0], device)
    model = finetune(data_loader, model, ns_epoch[1], learn_rates[1], criterion.loss_finetune, log_paths[1], state_paths[1], device)
    return model


def pretrain(data_loader, model, n_epoch, learn_rate, loss_func, log_path, state_path, device):
    """
    Pre-training process.

    Args:
        data_loader(torch.utils.data.DataLoader): Training data loader.
        model(model.FxInversion): Audio effect inversion Model.
        n_epoch(int): Number of epochs.
        learn_rate(float): Learing rate.
        log_paths(pathlib.Path): Path to the log file.
        state_paths(pathlib.Path): Path to the model state file.
        device(torch.device): Device on which the tensor calculation is performed.

    Returns:
        model(model.FxInversion): Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device == "cuda")
    for epoch in range(n_epoch):
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            dry_signal_origin, _, wet_signal, param = data
            dry_signal_origin, wet_signal, param = dry_signal_origin.to(device), wet_signal.to(device), param.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                dry_signal_est, param_est = model(wet_signal)
                mrstft, mse, loss = loss_func(dry_signal_origin, dry_signal_est, param, param_est)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            os.makedirs(log_path.parent, exist_ok=True)
            with open(log_path, "a") as log:
                log.write(f"{epoch}\t{batch_idx}\t{loss.item():.6f}\t{mrstft.item():.6f}\t{mse.item():.6f}\n")
    os.makedirs(state_path.parent, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    return model


def finetune(data_loader, model, n_epoch, learn_rate, loss_func, log_path, state_path, device):
    """
    Fine-tuning process.

    Args:
        data_loader(torch.utils.data.DataLoader): Training data loader.
        model(model.FxInversion): Audio effect inversion Model.
        n_epoch(int): Number of epochs.
        learn_rate(float): Learning rate.
        log_paths(pathlib.Path): Path to the log file.
        state_paths(pathlib.Path): Path to the model state file.
        device(torch.device): Device on which the tensor calculation is performed.

    Returns:
        model(model.FxInversion): Trained model.
    """
    for param in model.param_est.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device == "cuda")
    for epoch in range(n_epoch):
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            _, dry_signal_use, wet_signal, _ = data
            dry_signal_use, wet_signal = dry_signal_use.to(device), wet_signal.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                dry_signal_est, _ = model(wet_signal)
                loss = loss_func(dry_signal_use, dry_signal_est)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            os.makedirs(log_path.parent, exist_ok=True)
            with open(log_path, "a") as log:
                log.write(f"{epoch}\t{batch_idx}\t{loss.item():.6f}\n")
    os.makedirs(state_path.parent, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    return model
