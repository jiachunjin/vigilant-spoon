import cv2
import webdataset as wds
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


# class Dual_transform:
#     def __init__(self):
#         self.llamagen_transform = transforms.Compose([
#             transforms.Resize(256, max_size=None),
#             transforms.CenterCrop(256),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
#         ])
    
#     def build_tower(self, img):
#         img = np.array(img)
#         tower = build_gaussian_pyramid(img)
#         return tower

#     def __call__(self, img):
#         img1 = self.llamagen_transform(img)
#         img2 = self.build_tower(img)
#         return img1, img2


def get_dataloader(config):
    llamagen_transform = transforms.Compose([
        transforms.Resize(256, max_size=None),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dual_transform = Dual_transform()
    dataset = (
        wds.WebDataset(config.path, resampled=True, shardshuffle=False, nodesplitter=None)
        .shuffle(2048)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(llamagen_transform, None)
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=8, pin_memory=True)
    
    return dataloader