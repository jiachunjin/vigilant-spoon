# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model.lpips import LPIPS
from model.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from model.discriminator_stylegan import Discriminator as StyleGANDiscriminator



def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)

def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class Loss_entropy():
    def __init__(self, device):
        initial_value = 1.0
        decay_factor = 0.8
        # decay_factor = 0.5
        num_dims = 64

        factors = torch.ones(num_dims)
        factors[1:] = decay_factor 

        self.weight = torch.cumprod(factors, dim=0).to(device) * initial_value
        # self.weight = torch.cat([weight, torch.zeros_like(weight)], dim=0).to(device)

    @staticmethod
    def bernoulli_entropy(p):
        return -p * torch.log(torch.clip(p, 1e-10, 1)) - (1 - p) * torch.log(torch.clip(1 - p, 1e-10, 1))

    def __call__(self, bernoulli_p):
        # bernoulli_p: (b, 16, 16, 64)
        entropy = self.bernoulli_entropy(bernoulli_p).mean(dim=(0, 1, 2)) # (64,)

        return entropy * self.weight


class Loss_matryoshka():
    def __init__(self):
        pass

    def __call__(self, x, rec_matryoshka):
        loss = 0
        for rec in rec_matryoshka:
            loss += F.mse_loss(x, rec)
        
        return loss / len(rec_matryoshka)


class Loss_middle():
    def __init__(self):
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])
    
    def get_pyramids(self, batch, levels):
        pyramids = [] # b lists, each list contains a series of Gaussian Pyramid for each image
        pyramid_dict = dict(
            l1 = [],
            l2 = [],
            l4 = [],
            l8 = [],
            l16 = [],
            l32 = [],
            l64 = [],
            l128 = [],
        )
        for x in batch:
            x = self.inverse_transform(x)
            x = np.array(x)
            pyramid = build_gaussian_pyramid(x, levels=levels)

            pyramid_dict['l128'].append(pyramid[0])
            pyramid_dict['l64'].append(pyramid[1])
            pyramid_dict['l32'].append(pyramid[2])
            pyramid_dict['l16'].append(pyramid[3])
            pyramid_dict['l8'].append(pyramid[4])
            pyramid_dict['l4'].append(pyramid[5])
            pyramid_dict['l2'].append(pyramid[6])
            pyramid_dict['l1'].append(pyramid[7])
            # pyramids.append(pyramid)

        return pyramid_dict
    
    @staticmethod
    def list_of_numpy_to_tensor(numpy_arrays):
        # x is element of pyramid_dict
        torch_tensors = [torch.from_numpy(array) for array in numpy_arrays]
        torch_tensors = [tensor.float().permute(2, 0, 1) / 255.0 for tensor in torch_tensors]
        batch_tensor = torch.stack(torch_tensors)

        return batch_tensor
    
    def __call__(self, x, rec_intermediate):
        # x: (b, 3, 256, 256) (-1, 1)
        pyramid_dict = self.get_pyramids(x, levels=9)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l128'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_128 = F.mse_loss(rec_intermediate[0], target_tensors)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l64'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_64 = F.mse_loss(rec_intermediate[1], target_tensors)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l32'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_32 = F.mse_loss(rec_intermediate[2], target_tensors)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l16'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_16 = F.mse_loss(rec_intermediate[3], target_tensors)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l8'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_8 = F.mse_loss(rec_intermediate[4], target_tensors)

        target_tensors = self.list_of_numpy_to_tensor(pyramid_dict['l4'])
        target_tensors = target_tensors.to(x.device)
        target_tensors = target_tensors * 2 - 1
        loss_4 = F.mse_loss(rec_intermediate[5], target_tensors)

        loss = (loss_128 + loss_64 + loss_32 + loss_16 + loss_8 + loss_4) / 6

        return loss


        # rec_intermediate: dict, keys are l1, l2, l4, l8, l16, l32, l64, l128

def build_gaussian_pyramid(image, levels=8):
    """
    构建高斯金字塔
    :param image: 输入图像
    :param levels: 金字塔的层数
    :return: 高斯金字塔（包含所有层次）
    """
    pyramid = []
    # residuals = []
    
    for i in range(levels-1):
        # 使用高斯模糊然后下采样
        # print(image.shape)
        image = cv2.pyrDown(image)  # pyrDown用于下采样，自动做高斯模糊
        pyramid.append(image)
    
    return pyramid


class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, 
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, inputs, reconstructions, optimizer_idx, global_step, last_layer=None):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # discriminator loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            loss = self.rec_weight * rec_loss + \
                self.perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss
                # codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            # if global_step % log_every == 0:
            #     rec_loss = self.rec_weight * rec_loss
            #     p_loss = self.perceptual_weight * p_loss
            #     generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            #     logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
            #                 f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
            #                 f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
            #                 f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            
            # if global_step % log_every == 0:
            #     logits_real = logits_real.detach().mean()
            #     logits_fake = logits_fake.detach().mean()
            #     logger.info(f"(Discriminator) " 
            #                 f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
            #                 f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
            return d_adversarial_loss