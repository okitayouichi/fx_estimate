"""
Loss fuctions for training and evaluation.
"""

import config
import torch
import auraloss
import pedalboard as pb

loss_mae = torch.nn.L1Loss()
loss_mse = torch.nn.MSELoss()
loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss()


def loss_pretrain(dry_signal_origin, dry_signal_est, param, param_est, mrstft_weight=config.mrstft_weight):
    """
    Loss function for pretraining. Linear sum of MRSTFT and MSE loss.

    Args:
        dry_signal_origin(torch.Tensor): Ground truth dry signal. shape: (B, C, T)
        dry_signal_est(torch.Tensor): Estimated dry signal. shape: (B, C, T)
        param(torch.Tensor): Ground truth effect parameter. shape: (B, n_param)
        param_est(torch.Tensor): Estimated effect parameter. shape: (B, n_param)
        mrstft_weight(float): Weight of MRSTFT loss.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: MRSTFT loss.
            - torch.Tensor: MSE loss.
            - torch.Tensor: Linear sum of MRSTFT and MSE loss.
    """
    mrstft = loss_mrstft(dry_signal_origin, dry_signal_est)
    mse = loss_mse(param, param_est)
    loss = mrstft_weight * mrstft + mse
    return mrstft, mse, loss


def loss_finetune(dry_signal_use, dry_signal_est):
    """
    Loss function for fine-tuning. MRSTFT loss of dry signals.

    Args:
        dry_signal_use(torch.Tensor): Ground truth dry signal. shape: (B, C, T)
        dry_signal_est(torch.Tensor): Estimated dry signal. shape: (B, C, T)

    Returns:
        torch.Tensor: MRSTFT loss.
    """
    loss = loss_mrstft(dry_signal_use, dry_signal_est)
    return loss


def loss_reconst(wet_signal, dry_signal, fx, param):
    """
    Reconstruction loss for evaluation.
    Apply effect to dry signal and evaluate the reproducibility of the target wet signal.

    Args:
        wet_signal(torch.Tensor): Wet signal. shape: (1, C, T)
        dry_signal(torch.Tensor): Dry signal. shape: (1, C, T)
        fx(str): Audio effect name.
        param(torch.Tensor): Audio effect parameters. shape: (1, n_param)

    Returns:
        float: MRSTFT reconstruction loss.
    """
    param = param.squeeze(0).tolist()
    param = config.params_to_dict(fx, param)
    param = config.clip_param(fx, param)
    device = dry_signal.device
    dry_signal = norm_loudness(dry_signal).squeeze(0).cpu().detach().numpy()
    wet_signal_reconst = apply_fx(dry_signal, fx, param)
    wet_signal = norm_loudness(wet_signal)
    wet_signal_reconst = norm_loudness(torch.tensor(wet_signal_reconst).to(device).unsqueeze(0))
    loss = loss_mrstft(wet_signal, wet_signal_reconst).item()  # without grad
    return loss


def apply_fx(dry_signal, fx, param):
    """
    Apply audio effect.

    Args:
        dry_signal(numpy.array): Dry signal. shape: (C, T)
        fx(str): Audio effect name.
        param(dict): Audio effect parameter keywords.

    Returns:
        numpy.array: Wet signal. shape: (C, T)
    """
    plugin = config.fxs[fx]["plugin"]
    board = pb.Pedalboard([plugin(**param)])
    wet_signal = board(dry_signal, config.sample_rate)
    return wet_signal


def norm_loudness(signal, rms_target=0.1, eps=1.0e-8):
    """
    Normalize the loudness of an audio signal by the root-mean-square.

    Args:
        signal(torch.Tensor): Audio signal whose values range in [-1.0, 1.0). The last dimension is the time dimension.
        rms_target(float): Target value of root-mean-square.
        eps(float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Audio signal with normalized loudness.
    """
    rms = torch.sqrt(torch.mean(signal**2))
    rms = max(rms, eps)
    signal = signal / rms * rms_target

    return signal
