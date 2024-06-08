import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import rasterio

class MyDataSet(Dataset):
    def __init__(self, root_dir, image_paths):
        self.root_dir = root_dir
        self.image_paths = image_paths

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        img = np.array(Image.open(img_path)).astype(np.float32)
        img = img.reshape(1, 224, 224)
        # img = np.array(img, float)

        # img = Image.open(img_path)
        # img = np.array(img)
        # img = img.astype(np.float32)
        # img = img.reshape(1, 224, 224)
        # print(img.shape)

        label = int(self.image_paths[index].split('.')[-2])
        return img, label

    def __len__(self):
        return len(self.image_paths)


root_dir = "./data/standard/origin"
image_paths = os.listdir(root_dir)
train_size = int(image_paths.__len__() * 0.90)
train_images = image_paths[:train_size]
test_images = image_paths[train_size:]

train_set = MyDataSet(root_dir, train_images)
test_set = MyDataSet(root_dir, test_images)
test_set.__getitem__(0)

train_dataloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
