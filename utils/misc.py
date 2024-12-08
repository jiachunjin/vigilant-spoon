import torch
from torchvision import transforms
from einops import rearrange

def get_validation_img(img_path):
    from PIL import Image
    img = Image.open(img_path)
    llamagen_transform = transforms.Compose([
        transforms.Resize(256, max_size=None),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    img = llamagen_transform(img)

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage()
    ])
    
    return img, inverse_transform


def reconstrut_image(vqvae):
    vqvae.eval()
    device = next(vqvae.parameters()).device
    val_img, inverse_transform = get_validation_img('/home/jiachun/codebase/rfsq/assets/traffic.jpeg')
    val_img = val_img.unsqueeze(0).to(device)
    quant = vqvae.module.encode_binary(val_img)
    rec = vqvae.module.decode_fsq(quant)
    rec = inverse_transform(rec[0])

    return rec