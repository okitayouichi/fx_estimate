"""
Train and evaluate neural networks for audio effect estimation under different dry signal conditions. Models are built for each dry signal and effect type.
"""

import config
import dataset
import model
import train
import eval
import criterion
import torch
import sys

exp_num = sys.argv[1]
config.save_hypara(exp_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
for origin in config.origins:
    for fx in config.fxs.keys():
        torch.cuda.empty_cache()
        n_param = len(config.fxs[fx]["params"].keys())
        log_path_pt, log_path_ft = config.get_log_paths(exp_num, origin, fx)
        state_path_pt, state_path_ft = config.get_state_paths(exp_num, origin, fx)
        eval_path_pt, eval_path_ft = config.get_eval_paths(exp_num, origin, fx)

        # load dataset
        fx_dataset = dataset.FxDataset(config.use, origin, fx)
        length = len(fx_dataset)
        split_length = []
        for ratio in config.split_ratio:
            split_length.append(int(ratio * length))
        pt_dataset, ft_dataset, eval_dataset = torch.utils.data.random_split(fx_dataset, split_length, generator=torch.Generator().manual_seed(42))
        pt_loader = torch.utils.data.DataLoader(pt_dataset, batch_size=n_param * config.batch_size, shuffle=True, drop_last=True)
        ft_loader = torch.utils.data.DataLoader(ft_dataset, batch_size=n_param * config.batch_size, shuffle=True, drop_last=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, drop_last=True)

        # load model
        fx_inversion = model.FxInversion(n_param=n_param)
        fx_inversion.to(device)

        # pretrain
        train.pretrain(pt_loader, fx_inversion, config.n_epoch_pt, config.lr_pt, criterion.loss_pretrain, log_path_pt, state_path_pt, device)

        # evaluate
        eval.eval_pretrain(eval_loader, fx_inversion, fx, eval_path_pt, device)

        # finetune
        train.finetune(ft_loader, fx_inversion, config.n_epoch_ft, config.lr_ft, criterion.loss_finetune, log_path_ft, state_path_ft, device)

        # evaluate
        eval.eval_finetune(eval_loader, fx_inversion, fx, eval_path_ft, device)
