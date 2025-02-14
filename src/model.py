"""
Define the model architecture.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from demucs.spec import spectro, ispectro
from demucs.transformer import CrossTransformerEncoder, create_2d_sin_embedding
from demucs.hdemucs import pad1d, ScaledEmbedding, HEncLayer, HDecLayer
from demucs.demucs import rescale_module


class FxInversion(nn.Module):
    """
    Audio effect inversion model. Estimate the dry signal and effect parameter. This model is an extension of the Demucs.
    """

    def __init__(
        self,
        # channels
        num_channel_in=1,
        num_channel_hidden=24,
        growth=2,
        # STFT
        nfft=4096,
        # main structure
        layer_depth=3,
        # frequency embedding
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # convolutions
        kernel_size=8,
        stride=4,
        # dconv residual branch
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # transformer
        t_layers=2,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        t_cross_first=False,
        # weight init
        rescale=0.1,
        # parameter estimate branch
        p_depth_in_transformer=1,
        n_param=1,
        p_conv_depth=2,
        p_fc_depth=2,
        p_ch_rate_conv=2,
        p_ch_rate_fc=0.5,
        p_kernel=8,
        p_stride=4,
    ):
        super().__init__()
        self.layer_depth = layer_depth
        self.t_layers = t_layers
        self.p_depth_in_transformer = p_depth_in_transformer
        self.nfft = nfft
        self.hop_length = nfft // 4

        # time & frequency domain encoder & decoder
        self.encoders_t = nn.ModuleList()
        self.encoders_f = nn.ModuleList()
        self.decoders_t = nn.ModuleList()
        self.decoders_f = nn.ModuleList()
        chin_t = num_channel_in
        chin_f = 2 * num_channel_in
        chout = num_channel_hidden
        for depth in range(layer_depth):
            kw = {
                "kernel_size": kernel_size,
                "stride": stride,
                "norm": False,
                "dconv_kw": {"depth": dconv_depth, "compress": dconv_comp, "init": dconv_init, "gelu": True},
            }
            encoder_t = HEncLayer(chin=chin_t, chout=chout, freq=False, dconv=True, **kw)
            encoder_f = HEncLayer(chin=chin_f, chout=chout, freq=True, dconv=True, **kw)
            decoder_t = HDecLayer(chin=chout, chout=chin_t, last=depth == 0, freq=False, dconv=False, **kw)
            decoder_f = HDecLayer(chin=chout, chout=chin_f, last=depth == 0, freq=True, dconv=False, **kw)
            self.encoders_t.append(encoder_t)
            self.encoders_f.append(encoder_f)
            self.decoders_t.append(decoder_t)
            self.decoders_f.append(decoder_f)
            chin_t = chout
            chin_f = chout
            chout *= growth
            if depth == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding((nfft // 2) // stride, chin_f, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        # initialize convolution weight
        if rescale:
            rescale_module(self, reference=rescale)

        # transformer encoder
        transformer_channels = num_channel_hidden * growth**depth
        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )

        # parameter estimate branch
        self.param_est = FxParamEstimate(n_param, transformer_channels, p_conv_depth, p_fc_depth, p_ch_rate_conv, p_ch_rate_fc, p_kernel, p_stride)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _complex_to_channel(self, x):
        x = torch.view_as_real(x)  # (B, C, F, T, Cmp)
        x = rearrange(x, "b c f t cmp -> b (c cmp) f t")
        return x

    def _channel_to_complex(self, x):
        x = rearrange(x, "b (c cmp) f t -> b c f t cmp", cmp=2).contiguous()
        x = torch.view_as_complex(x)
        return x

    def forward(self, x):
        """
        Args:
            x(torch.Tensor): Wet audio signal. shape: (B, C, T)

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Estimated dry audio signal. shape: (B, C, T)
                - torch.Tensor: Estimated effect parameter. shape: (B, n_param)
        """
        length = x.shape[-1]
        x_t = x
        x_f = self._spec(x)
        x_f = self._complex_to_channel(x_f)

        skips_t = []  # skips_t[i] is the output of the i-th decoder
        skips_f = []
        lengths_t = []  # lengths_t[i] is the input time length of the i-th decoder
        lengths_f = []

        # time & frequency domain encoder
        for depth in range(self.layer_depth):
            lengths_t.append(x_t.shape[-1])
            lengths_f.append(x_f.shape[-1])
            encoder_t = self.encoders_t[depth]
            encoder_f = self.encoders_f[depth]
            x_t = encoder_t(x_t)
            x_f = encoder_f(x_f)
            if depth == 0 and self.freq_emb is not None:
                frs = torch.arange(x_f.shape[-2], device=x_f.device)
                emb = rearrange(self.freq_emb(frs), "f c -> 1 c f 1").expand_as(x_f)
                x_f = x_f + self.freq_emb_scale * emb
            skips_t.append(x_t)
            skips_f.append(x_f)

        # cross-domain transformer encoder
        B, C, T1 = x_t.shape
        x_t = rearrange(x_t, "b c t2 -> b t2 c")  # C-dim is the last for transformer
        pos_emb = self.crosstransformer._get_pos_embedding(T1, B, C, x_t.device)
        pos_emb = rearrange(pos_emb, "t1 b c -> b t1 c")
        x_t = self.crosstransformer.norm_in_t(x_t)
        x_t = x_t + self.crosstransformer.weight_pos_embed * pos_emb
        B, C, Fr, T2 = x_f.shape
        pos_emb_2d = create_2d_sin_embedding(C, Fr, T2, x_f.device, self.crosstransformer.max_period)  # (1, C, Fr, T2)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t2 -> b (t2 fr) c")
        x_f = rearrange(x_f, "b c fr t2 -> b (t2 fr) c")  # C-dim is the last for transformer
        x_f = self.crosstransformer.norm_in(x_f)
        x_f = x_f + self.crosstransformer.weight_pos_embed * pos_emb_2d
        for depth in range(self.crosstransformer.num_layers):
            if depth % 2 == self.crosstransformer.classic_parity:
                x_f = self.crosstransformer.layers[depth](x_f)
                x_t = self.crosstransformer.layers_t[depth](x_t)
            else:
                old_x_f = x_f
                x_f = self.crosstransformer.layers[depth](x_f, x_t)
                x_t = self.crosstransformer.layers_t[depth](x_t, old_x_f)
            # config estimate beanch
            if depth == self.p_depth_in_transformer:
                x_t_ = rearrange(x_t, "b t1 c -> b c t1")
                x_f_ = rearrange(x_f, "b frt2 c -> b c frt2")
                fx_param = self.param_est(x_t_, x_f_)
        x_t = rearrange(x_t, "b t1 c -> b c t1")
        x_f = rearrange(x_f, "b (t2 fr) c -> b c fr t2", t2=T2)

        # time & frequency domain decoder
        for depth in range(self.layer_depth - 1, -1, -1):
            length_t = lengths_t.pop()
            length_f = lengths_f.pop()
            skip_f = skips_f.pop()
            skip_t = skips_t.pop(-1)
            decoder_t = self.decoders_t[depth]
            decoder_f = self.decoders_f[depth]
            x_t, _ = decoder_t(x_t, skip_t, length_t)
            x_f, _ = decoder_f(x_f, skip_f, length_f)

        x_f = self._channel_to_complex(x_f)
        x_f = self._ispec(x_f, length)
        x = x_t + x_f

        return x, fx_param


class FxParamEstimate(nn.Module):
    """
    Effect parameter estimation branch for FxInversion. This is a CNN-based model.
    """

    def __init__(self, n_param, chs_in, conv_depth=2, fc_depth=2, ch_rate_conv=2, ch_rate_fc=0.5, kernel=8, stride=4):
        """
        Args:
            n_param(int): Number of the effect parameters.
            chs_in(int): Number of channels of input audio signal.
            conv_depth(int): Number of convolution blocks. The same number of blocks is constructed for time and frequency domain.
            fc_depth(int): Number of fully-connected blocks.
            ch_rate_conv(int): Rate of number of channels in convolution blocks.
            ch_rate_fc(int): Rate of number of channels in fully-connected blocks.
            kernel(int): Kernel size for convolution layers.
            stride(int): Stride for convolution layers.
        """
        super().__init__()

        # convolution blocks
        chs_in = chs_in
        chs_out = chs_in * ch_rate_conv
        self.conv_depth = conv_depth
        self.convs_t = nn.ModuleList()
        self.convs_f = nn.ModuleList()
        for depth in range(self.conv_depth):
            conv_t = nn.Sequential(nn.Conv1d(in_channels=chs_in, out_channels=chs_out, kernel_size=kernel, stride=stride), nn.BatchNorm1d(num_features=chs_out), nn.ReLU())
            conv_f = nn.Sequential(nn.Conv1d(in_channels=chs_in, out_channels=chs_out, kernel_size=kernel, stride=stride), nn.BatchNorm1d(num_features=chs_out), nn.ReLU())
            self.convs_t.append(conv_t)
            self.convs_f.append(conv_f)
            chs_in = chs_out
            chs_out *= ch_rate_conv

        # fully-connceted blocks
        chs_in *= 2
        chs_out = int(chs_in * ch_rate_fc)
        self.fc_depth = fc_depth
        self.fcs = nn.ModuleList()
        for depth in range(self.fc_depth):
            fc = nn.Sequential(nn.Linear(in_features=chs_in, out_features=chs_out), nn.BatchNorm1d(num_features=chs_out), nn.ReLU(), nn.Dropout(p=0.05))
            self.fcs.append(fc)
            chs_in = chs_out
            if depth < fc_depth - 2:
                chs_out = int(chs_out * ch_rate_fc)
            else:
                chs_out = n_param

    def _pool(self, x):
        """
        Pool by adding the average and maximum along the time axis.

        Args:
            x(torch.Tensor): Time series feature. shape: (B, C, T)

        Returns:
            torch.Tensor: Pooled feature. shape: (B, C)
        """
        mean = torch.mean(x, dim=-1)
        max = torch.max(x, dim=-1).values
        return mean + max

    def forward(self, x_t, x_f):
        """
        Args:
            x_t(torch.Tensor): Time domain feature from cross-domain transformer in FxInversion. shape: (B, C, T1)
            x_f(torch.Tensor): Freqency domain feature from cross-domain transformer in FxInversion. shape: (B, C, F*T2)

        Returns:
            torch.Tensor: Estimated effect parameter. shape: (B, n_param)
        """
        for depth in range(self.conv_depth):
            conv_t = self.convs_t[depth]
            conv_f = self.convs_f[depth]
            x_t = conv_t(x_t)
            x_f = conv_f(x_f)
        x_t = self._pool(x_t)
        x_f = self._pool(x_f)
        x = torch.cat([x_t, x_f], dim=-1)
        for depth in range(self.fc_depth):
            fc = self.fcs[depth]
            x = fc(x)
        return x
