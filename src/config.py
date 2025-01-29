"""
Settings and functions related to the overall project configuration, including datasets.
"""

from pathlib import Path
import pedalboard as pb
import os
from dotenv import load_dotenv

# path
load_dotenv()
project_path = Path(os.getenv("PROJECT_PATH"))  # path to the audio effect estimation project
dry_signal_dataset_path = project_path / "gt_dataset"  # path to the dry signal dataset
wet_signal_dataset_path = project_path / "gtfx_dataset"  # path to the wet signal dataset
fx_estimate_path = project_path / "fx_estimate"  # path to the audio effect deep learning project

# data
sample_rate = 44100
insts = ["sc", "tc", "lp"]
use = "sc"
origins = [inst for inst in insts if inst != use]
strings = range(1, 7)
frets = range(20)
num_fx_grid = 32
fxs = {
    "distortion": {"plugin": pb.Distortion, "params": {"drive_db": [10.0, 50.0]}},
    "reverb": {"plugin": pb.Reverb, "params": {"room_size": [0.1, 1.0]}},
}

# train
train_ratio = 0.8
batch_size = 32
ns_epoch = (50, 50)
learn_rates = (1.0e-3, 5.0e-4)
mrstft_weight = 0.2


def save_hypara(exp_num):
    path = fx_estimate_path / "result" / ("hypara" + str(exp_num).zfill(3) + ".log")
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{train_ratio}\t{batch_size}\t{ns_epoch}\t{learn_rates}\t{mrstft_weight}\n")


def get_hypara_path(exp_num):
    dir_path = fx_estimate_path / "result"
    path = dir_path / "hypara" + str(exp_num).zfill(3) + ".log"
    return path


def get_log_paths(exp_num, origin, fx):
    dir_path = fx_estimate_path / "result" / origin / fx
    pretrain_file_name = "pretrain" + str(exp_num).zfill(3) + ".log"
    finetune_file_name = "finetune" + str(exp_num).zfill(3) + ".log"
    paths = (dir_path / pretrain_file_name, dir_path / finetune_file_name)
    return paths


def get_state_paths(exp_num, origin, fx):
    dir_path = fx_estimate_path / "result" / origin / fx
    pretrain_file_name = "pretrain" + str(exp_num).zfill(3) + ".pth"
    finetune_file_name = "finetune" + str(exp_num).zfill(3) + ".pth"
    paths = (dir_path / pretrain_file_name, dir_path / finetune_file_name)
    return paths


def get_eval_path(exp_num, origin, fx):
    dir_path = fx_estimate_path / "result" / origin / fx
    file_name = "eval" + str(exp_num).zfill(3) + ".log"
    path = dir_path / file_name
    return path


def get_wet_nums():
    """
    Generate the dictionary of the data number in "gtfx_dataset".

    Returns:
        wet_nums(dict): The dictionary of the data number.
    """
    # initialize dict
    wet_nums = {}
    for inst in insts:
        wet_nums[inst] = {}
        for fx in fxs.keys():
            wet_nums[inst][fx] = []
    # append data number
    wet_num = 0
    for inst in insts:
        for string in strings:
            for fret in frets:
                for fx in fxs.keys():
                    for i in range(num_fx_grid ** len(fxs[fx]["params"].keys())):
                        wet_nums[inst][fx].append(wet_num)
                        wet_num += 1
    return wet_nums


def get_signal_path(data_num, inst, fx=None):
    """
    Get the path to the audio file in "gt_dataset" or "gtfx_dataset".

    Args:
        data_num(int): Data number.
        inst(str): Instrument name.
        fx(str): Audio effect name.

    Returns:
        file_path(pathlib.Path): Path to the audio file.
    """
    if fx is None:
        dir_path = dry_signal_dataset_path / "data" / inst / "audio"
        file_name = "gt" + str(data_num).zfill(8) + ".flac"
        file_path = dir_path / file_name
    else:
        dir_path = wet_signal_dataset_path / "data" / inst / fx / "audio"
        file_name = "gtfx" + str(data_num).zfill(8) + ".flac"
        file_path = dir_path / file_name
    return file_path


def get_label_path(data_num, inst, fx):
    """
    Generate the path to the label file in "gtfx_dataset".

    Args:
        data_num(int): Data number.
        inst(str): Instrument name.
        fx(str): Audio effect name.

    Returns:
        file_path(pathlib.Path): Path to the label file.
    """
    dir_path = wet_signal_dataset_path / "data" / inst / fx / "label"
    file_name = "gtfx" + str(data_num).zfill(8) + ".json"
    file_path = dir_path / file_name
    return file_path


def label_to_params(label):
    """
    Convert the dictionary of labels of wet signal and normalize them in [0, 1].

    Args:
        label(dict): Label of wet signal.

    Returns:
        params(list): Audio effect parameters.
    """
    params = []
    fx = label["fx"]["type"]
    for param_name in fxs[fx]["params"].keys():
        min = fxs[fx]["params"][param_name][0]
        max = fxs[fx]["params"][param_name][1]
        param = label["fx"]["params"][param_name]
        param = (param - min) / (max - min)
        params.append(param)
    return params


def params_to_dict(fx, params):
    """
    Restore normalized effect parameters and convert to a dictionary of effect parameters.

    Args:
        fx(str): Audio effect name.
        params(list): Audio effect parameters.

    Returns:
        param_dict(dict): Dictionary of effect parameters.
    """
    param_dict = {}
    for i, param_name in enumerate(fxs[fx]["params"].keys()):
        min = fxs[fx]["params"][param_name][0]
        max = fxs[fx]["params"][param_name][1]
        param = params[i] * (max - min) + min
        param_dict[param_name] = param
    return param_dict


def clip_param(fx, param_dict):
    """
    Clip audio effect parameter value for expected range.

    Args:
        param_dict(dict): Dictionary of effect parameters.

    Returns:
        param_dict(dict): Dictionary of effect parameters which value is clipped.
    """
    for param_name in param_dict.keys():
        param_val = param_dict[param_name]
        min = fxs[fx]["params"][param_name][0]
        max = fxs[fx]["params"][param_name][1]
        if param_val < min:
            param_val = min
        elif param_val > max:
            param_val = max
        param_dict[param_name] = param_val
    return param_dict


def wet_num_to_dry_num(wet_num):
    """
    From the data number of the wet signal, obtain the corresponding the data number of the dry signal for using and the original dry signal used to generate it.

    Args:
        wet_num(int): Data number of the wet signal.

    Returns:
        dry_use_num(int): Data number of the dry signal for using.
        dry_origin_num(int): Data number of the dry signal used to generate the wet signal.
    """
    n_dry_per_inst = len(strings) * len(frets)
    n_wet_per_dry = 0
    for fx in fxs.keys():
        n_wet_per_dry += num_fx_grid ** len(fxs[fx]["params"].keys())
    dry_origin_num = wet_num // n_wet_per_dry
    dry_use_num = dry_origin_num % n_dry_per_inst
    return dry_use_num, dry_origin_num
