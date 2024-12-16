# 检查validation set中的bae的平均Bernoulli p以及entropy
import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.vq_llamagen import VQModel

preprocess = transforms.Compose([
    transforms.Resize(256, max_size=None),  # 将短边调整为 256
    transforms.CenterCrop(256),  # 中心裁剪为 224x224
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

config = OmegaConf.create({
    'codebook_dim': 64,
    'encoder_ch_mult': [1, 1, 2, 2, 4],
    'decoder_ch_mult': [1, 1, 2, 2, 4],
    'bernoulli': True,
    'use_negative': False,
    'z_channels': 256,
    'dropout_p': 0.0,
    'matryoshka': True,
})
device = torch.device('cuda:7')

bae = VQModel(config)
ckpt = torch.load('ckpts/a800/vqvae-matryoshka_64-170k', map_location='cpu')
bae.load_state_dict(ckpt, strict=False)
bae.eval()
bae = bae.to(device)

val_dataset = ImageNet(root='/data/Largedata/ImageNet', split='val', transform=preprocess)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=1)

print('done')
def bernoulli_entropy(p):
    # 避免 log(0) 的情况，使用 np.where 处理边界
    return -p * torch.log(torch.clip(p, 1e-10, 1)) - (1 - p) * torch.log(torch.clip(1 - p, 1e-10, 1))

entropy = 0
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        quant, p = bae.encode_binary(images)
        entropy = bernoulli_entropy(p).mean(dim=[0, 1, 2])
        print(entropy.shape)
        exit(0)
        # entropy += entropy.mean().item()

# entropy /= 500
# print(entropy)