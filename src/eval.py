"""
Evaluation process.
"""

import criterion
import torch
import os


def eval(data_loader, model, fx, path, device):
    """
    Evaluation process.

    Args:
        data_loader(torch.utils.data.DataLoader): Evaluation data loader.
        model(model.FxInversion): Audio effect inversion Model.
        path(pathlib.Path): Path to the log file.
        device(torch.device): Device on which the tensor calculation is performed.
    """
    model.eval()
    for data_idx, data in enumerate(data_loader):
        with torch.no_grad():
            _, dry_signal_use, wet_signal, param = data
            dry_signal_use, wet_signal, param = dry_signal_use.to(device), wet_signal.to(device), param.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                dry_signal_est, param_est = model(wet_signal)
                dry_loss = criterion.loss_finetune(dry_signal_use, dry_signal_est).item()
                reconst_dry_true_param_origin = criterion.loss_reconst(wet_signal, dry_signal_use, fx, param)
                reconst_dry_true = criterion.loss_reconst(wet_signal, dry_signal_use, fx, param_est)
                reconst_dry_est_param_origin = criterion.loss_reconst(wet_signal, dry_signal_est, fx, param)
                reconst_dry_est = criterion.loss_reconst(wet_signal, dry_signal_est, fx, param_est)
            os.makedirs(path.parent, exist_ok=True)
            with open(path, "a") as output:
                output.write(f"{data_idx}\t{dry_loss}\t{reconst_dry_true_param_origin:.6f}\t{reconst_dry_true:.6f}\t{reconst_dry_est_param_origin:.6f}\t{reconst_dry_est:.6f}\n")
