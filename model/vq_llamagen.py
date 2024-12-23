# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bernoulli = config.bernoulli
        self.use_negative = config.use_negative
        self.matryoshka = config.matryoshka

        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quant_conv_new = nn.Conv2d(config.z_channels, config.codebook_dim, 1)
        self.post_quant_conv_new = nn.Conv2d(config.codebook_dim, config.z_channels, 1)

        # used only when matryoshka
        if self.matryoshka:
            self.post_qc_48 = nn.Conv2d(48, config.z_channels, 1) # 128
            self.post_qc_32 = nn.Conv2d(32, config.z_channels, 1) # 64
            self.post_qc_24 = nn.Conv2d(24, config.z_channels, 1) # 32
            self.post_qc_16 = nn.Conv2d(16, config.z_channels, 1) # 16
            self.post_qc_8 = nn.Conv2d(8, config.z_channels, 1) # 8
            self.post_qc_4 = nn.Conv2d(4, config.z_channels, 1) # 4

            self.decoder_128 = Decoder(ch=32, ch_mult=[2, 2, 2, 2],z_channels=config.z_channels, out_channels=3)
            self.decoder_64 = Decoder(ch=32, ch_mult=[2, 2, 2],z_channels=config.z_channels, out_channels=3)
            self.decoder_32 = Decoder(ch=32, ch_mult=[2, 2],z_channels=config.z_channels, out_channels=3)
            self.decoder_16 = Decoder(ch=32, ch_mult=[2],z_channels=config.z_channels, out_channels=3)
            self.decoder_8 = Decoder(ch=32, ch_mult=[2, 2],z_channels=config.z_channels, out_channels=3, downsample=2)
            self.decoder_4 = Decoder(ch=32, ch_mult=[2, 2],z_channels=config.z_channels, out_channels=3, downsample=4)
        else:
            raise NotImplementedError

    def encode_binary(self, x):
        h = self.encoder(x)
        h = self.quant_conv_new(h)
        
        h = rearrange(h, 'b c h w -> b h w c')
        # move to [-1, 1]
        if self.use_negative:
            h = torch.nn.functional.tanh(h)
            if self.bernoulli:
                h_hat = torch.bernoulli((h + 1) / 2).to(h.dtype)
                h_hat = h_hat * 2 - 1
            else:
                h_hat = (h > 0.0).to(h.dtype)
                h_hat = h_hat * 2 - 1
        else:
            h = torch.nn.functional.sigmoid(h)
            if self.bernoulli:
                h_hat = torch.bernoulli(h).to(h.dtype)
            else:
                h_hat = (h > 0.5).to(h.dtype)

        quant = h + (h_hat - h).detach()
        quant = rearrange(quant, 'b h w c -> b c h w')

        return quant, h

    def decode_fsq(self, quant):
        quant = self.post_quant_conv_new(quant)
        dec = self.decoder(quant)
        return dec
    
    def decode_intermediate(self, quant):
        assert self.matryoshka == True, "matryoshka must be true"
        quant_48 = self.post_qc_48(quant[:, :48, :, :])
        dec_128 = self.decoder_128(quant_48)

        quant_32 = self.post_qc_32(quant[:, :32, :, :])
        dec_64 = self.decoder_64(quant_32)

        quant_24 = self.post_qc_24(quant[:, :24, :, :])
        dec_32 = self.decoder_32(quant_24)

        quant_16 = self.post_qc_16(quant[:, :16, :, :])
        dec_16 = self.decoder_16(quant_16)

        quant_8 = self.post_qc_8(quant[:, :8, :, :])
        dec_8 = self.decoder_8(quant_8)

        quant_4 = self.post_qc_4(quant[:, :4, :, :])
        dec_4 = self.decoder_4(quant_4)

        return (dec_128, dec_64, dec_32, dec_16, dec_8, dec_4)

    def get_bernoulli(self, x):
        h = self.encoder(x)
        h = self.quant_conv_new(h)
        h = rearrange(h, 'b c h w -> b h w c')
        h = torch.nn.functional.sigmoid(h).detach()

        return h
    
    @torch.no_grad()
    def encode_for_gpt(self, x):
        h = self.encoder(x)
        h = self.quant_conv_new(h)
        
        h = rearrange(h, 'b c h w -> b h w c')
        # move to [-1, 1]
        if self.use_negative:
            raise NotImplementedError
            h = torch.nn.functional.tanh(h)
            if self.bernoulli:
                h_hat = torch.bernoulli((h + 1) / 2).to(h.dtype)
                h_hat = h_hat * 2 - 1
            else:
                h_hat = (h > 0.0).to(h.dtype)
                h_hat = h_hat * 2 - 1
        else:
            h = torch.nn.functional.sigmoid(h)
            if self.bernoulli:
                binary_samples = torch.bernoulli(h).to(h.dtype)
            else:
                raise NotImplementedError
        
        binary_samples = rearrange(binary_samples, 'b h w c -> b (h w) c')
        h = rearrange(h, 'b h w c -> b (h w) c')
        return binary_samples, h


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3, downsample=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                if downsample is None:
                    conv_block.upsample = Upsample(block_in, resamp_with_conv)
                elif downsample == 2:
                    conv_block.upsample = Downsample_2(block_in, resamp_with_conv)
                elif downsample == 4:
                    conv_block.upsample = Downsample_4(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample_2(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample_4(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.25, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
