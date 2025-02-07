"""
Evaluation process.
"""

import criterion
import torch
import os


def eval_pretrain(data_loader, model, fx, path, device):
    """
    Evaluation process after pre-training before fine-tuning.

    Args:
        data_loader(torch.utils.data.DataLoader): Evaluation data loader.
        model(model.FxInversion): Audio effect inversion Model.
        path(pathlib.Path): Path to the log file.
        device(torch.device): Device on which the tensor calculation is performed.
    """
    model.eval()
    for data_idx, data in enumerate(data_loader):
        with torch.no_grad():
            wet_num, dry_signal_origin, dry_signal_use, wet_signal, param = data
            dry_signal_origin, dry_signal_use, wet_signal, param = dry_signal_origin.to(device), dry_signal_use.to(device), wet_signal.to(device), param.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                wet_num = wet_num.item()
                dry_signal_est, param_est = model(wet_signal)
                mae = criterion.loss_mae(param, param_est).item()
                mrstft = criterion.loss_mrstft(dry_signal_origin, dry_signal_est).item()
                mrstft_use = criterion.loss_mrstft(dry_signal_use, dry_signal_est).item()
                reconst_dry_true = criterion.loss_reconst(wet_signal, dry_signal_origin, fx, param_est)
                reconst_dry_est = criterion.loss_reconst(wet_signal, dry_signal_est, fx, param_est)
                reconst_dry_use = criterion.loss_reconst(wet_signal, dry_signal_use, fx, param_est)
            os.makedirs(path.parent, exist_ok=True)
            with open(path, "a") as output:
                output.write(f"{data_idx:4d}\t")
                output.write(f"{wet_num:6d}\t")
                output.write(f"{mae:.6f}\t")
                output.write(f"{mrstft:.6f}\t")
                output.write(f"{mrstft_use:.6f}\t")
                output.write(f"{reconst_dry_true:.6f}\t")
                output.write(f"{reconst_dry_est:.6f}\t")
                output.write(f"{reconst_dry_use:.6f}\t")
                output.write("\n")


def eval_finetune(data_loader, model, fx, path, device):
    """
    Evaluation process after fine-tuning.

    Args:
        data_loader(torch.utils.data.DataLoader): Evaluation data loader.
        model(model.FxInversion): Audio effect inversion Model.
        path(pathlib.Path): Path to the log file.
        device(torch.device): Device on which the tensor calculation is performed.
    """
    model.eval()
    for data_idx, data in enumerate(data_loader):
        with torch.no_grad():
            wet_num, dry_signal_origin, dry_signal_use, wet_signal, param = data
            dry_signal_origin, dry_signal_use, wet_signal, param = dry_signal_origin.to(device), dry_signal_use.to(device), wet_signal.to(device), param.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                wet_num = wet_num.item()
                dry_signal_est, param_est = model(wet_signal)
                mrstft = criterion.loss_mrstft(dry_signal_use, dry_signal_est).item()
                mrstft_origin = criterion.loss_mrstft(dry_signal_origin, dry_signal_est).item()
                reconst_dry_true_param_origin = criterion.loss_reconst(wet_signal, dry_signal_use, fx, param)
                reconst_dry_true = criterion.loss_reconst(wet_signal, dry_signal_use, fx, param_est)
                reconst_dry_est_param_origin = criterion.loss_reconst(wet_signal, dry_signal_est, fx, param)
                reconst_dry_est = criterion.loss_reconst(wet_signal, dry_signal_est, fx, param_est)
            os.makedirs(path.parent, exist_ok=True)
            with open(path, "a") as output:
                output.write(f"{data_idx:4d}\t")
                output.write(f"{wet_num:6d}\t")
                output.write(f"{mrstft:.6f}\t")
                output.write(f"{mrstft_origin:.6f}\t")
                output.write(f"{reconst_dry_true_param_origin:.6f}\t")
                output.write(f"{reconst_dry_true:.6f}\t")
                output.write(f"{reconst_dry_est_param_origin:.6f}\t")
                output.write(f"{reconst_dry_est:.6f}\t")
                output.write("\n")
