"""
Train and evaluate neural networks for audio effect estimation under different dry signal conditions. Models are built for each dry signal and effect type.
"""

import config
import dataset
import model
import train
import eval
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

        # load dataset
        fx_dataset = dataset.FxDataset(config.use, origin, fx)
        length = len(fx_dataset)
        train_length = int(config.train_ratio * length)
        split_len = [train_length, length - train_length]
        train_dataset, eval_dataset = torch.utils.data.random_split(fx_dataset, split_len, generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_param * config.batch_size, shuffle=True, drop_last=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, drop_last=True)

        # load model
        fx_inversion = model.FxInversion(n_param=n_param)
        fx_inversion.to(device)

        # train
        log_paths = config.get_log_paths(exp_num, origin, fx)
        state_paths = config.get_state_paths(exp_num, origin, fx)
        fx_inversion = train.train(train_loader, fx_inversion, config.ns_epoch, config.learn_rates, log_paths, state_paths, device)

        # evaluate
        eval_path = config.get_eval_path(exp_num, origin, fx)
        eval.eval(eval_loader, fx_inversion, fx, eval_path, device)
