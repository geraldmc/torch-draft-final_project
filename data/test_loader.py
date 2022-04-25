import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from data.transforms import data_transforms

# See: https://debuggercafe.com/custom-dataset-and-dataloader-in-pytorch/

class DeepWeeds_Test(Dataset):

    def __init__(self, csv_file, test=True):
        """
        """
        self.sub_directory = 'data/test'
        self.csv_file_name = csv_file
        self.transform = data_transforms['default']
        self.frame = None

        csv_file_path = os.path.join(self.root_dir, 
                                     self.sub_directory, 
                                     self.csv_file_name)
        
        self.csv_data = pd.read_csv(csv_file_path, sep=',')

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 
                                self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img