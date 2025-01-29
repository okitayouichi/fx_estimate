"""
Dataset class.
"""

import config
import torch
import torchaudio
import json

wet_nums = config.get_wet_nums()


class FxDataset(torch.utils.data.Dataset):
    """
    Dataset for audio effect estimation under different dry signal conditions.
    """

    def __init__(self, use, origin, fx):
        super().__init__()
        self.use = use
        self.origin = origin
        self.fx = fx
        self.wet_nums = wet_nums[origin][fx]
        params = []
        for wet_num in self.wet_nums:
            path = config.get_label_path(wet_num, origin, fx)
            with open(path, "r") as f:
                label = json.load(f)
            param = config.label_to_params(label)
            params.append(param)
        params = torch.tensor(params)
        self.params = params

    def __getitem__(self, index):
        wet_num = self.wet_nums[index]
        dry_use_num, dry_origin_num = config.wet_num_to_dry_num(wet_num)
        dry_signal_use_path = config.get_signal_path(dry_use_num, self.use)
        dry_signal_origin_path = config.get_signal_path(dry_origin_num, self.origin)
        wet_signal_path = config.get_signal_path(wet_num, self.origin, self.fx)
        dry_signal_use, _ = torchaudio.load(str(dry_signal_use_path))
        dry_signal_origin, _ = torchaudio.load(str(dry_signal_origin_path))
        wet_signal, _ = torchaudio.load(str(wet_signal_path))
        param = self.params[index]
        return dry_signal_origin, dry_signal_use, wet_signal, param

    def __len__(self):
        length = len(self.wet_nums)
        return length
