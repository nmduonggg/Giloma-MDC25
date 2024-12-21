import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
from data import transforms

class IndependentPatchDataset(Dataset):
    def __init__(self, image_dir, data_path, mode):
        super().__init__()
        assert(mode in ['training', 'testing', 'valid'])
        self.mode = mode
        if mode=='valid': self.mode='training'
        self.image_dir = image_dir
        self.data_list = json.load(open(data_path))
        
        self.label2idx = {
            'Mitosis': 1, 'Non-mitosis': 0
        }
        
        if mode=='training': self.transforms = transforms.TRAIN_TRANSFORMS
        else: self.transforms = transforms.TEST_TRANSFORMS
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path = os.path.join(self.image_dir, f"{self.mode}_{data['id']}.jpg")
        
        img = Image.open(img_path).convert("RGB")
        
        x = self.transforms(img)
        if self.mode=='testing':
            return x, data
        
        cls_num = self.label2idx[data['label']]
        y = torch.tensor(cls_num).long()
        
        return x, y
        
        