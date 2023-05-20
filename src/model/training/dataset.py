import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class FashionDataset(Dataset):
    '''
    This class is used to load the data for fashion classification.
    '''
    def __init__(self, img_list:List[str], label_list: List[int], data_dir:Path, transform:transforms = None, debug: bool = False):
        '''
        self.img_list:
        self.label_list:
        self.transform:
        self.data_dir:
        '''
        self.img_list = img_list
        self.label_list = label_list
        assert len(self.img_list) == len(label_list)
        self.transform = transform
        self.data_dir = data_dir
        self.debug = debug
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        '''
        idx: index of the item
        return: tuple of (x, y), where x is the input and y is the label.
        '''
        image_filepath = f"{self.data_dir}/{self.img_list[idx]}"
        image = Image.open(image_filepath).convert('RGB')
        label = self.label_list[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        if self.debug:
            return image, label, self.img_list[idx]
        else:
            return image, label