from configparser import Interpolation
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets

import numpy as np
#import matplotlib.pyplot as plt
import cv2

import glob
import random
import time
import os


class MyDataset(Dataset):
    def __init__(self, images, transform=False):
        '''
        MyDataset params
        images_psnrs(list): [(image1_path, image2_path, psnr_sum), ...]
        '''
        super(MyDataset, self).__init__()
        self.images = images
        self.transform = transform
        
    def __getitem__(self, index):
        image = self.images[index]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

def get_file_list(src_dir, target_list):
    print("get_file_list")

def main():
    # dir = "/dev/shm/dataset/n01930112"
    dir = "../../n01930112"
    image_dir = os.path.join(dir,"*.JPEG")
    images = sorted(glob.glob(image_dir))
    
    i = 0
    print(images[0])
    print(len(images))
    
    start_at = time.time()
    data_transform = torchvision.transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((240, 240)),
                                                    transforms.ColorJitter(contrast=0.5, saturation=0.5),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()
                                                    ])
    now_dataset = MyDataset(images, transform=data_transform)
    dataset_at = time.time()

    train_loader = DataLoader(now_dataset, batch_size=1,
                          num_workers=0, drop_last=False, shuffle=True, pin_memory=True)
    loader_at = time.time()

    dataset_time = (dataset_at-start_at)*1000
    loader_time = (loader_at-dataset_at)*1000

    print(f"{dataset_time:.4f} {loader_time:.4f}")

    loop_at = time.time()
    for image in train_loader:
        unit_at = time.time()
        diff_time = (unit_at-loop_at)*1000
        print(f"{diff_time:.4f}")
        loop_at = time.time()

if __name__ == "__main__":
    # run main program
    main()
