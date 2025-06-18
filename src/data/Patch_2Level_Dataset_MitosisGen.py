import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image, ImageFilter
from data import transforms
import torchvision.transforms as vis_transforms
import random
from labelme import utils as lbl_utils

class Patch_2Level_Dataset_MitosisGen(Dataset):
    def __init__(self, image_dir, data_path, mode, image_dir2=None, data_path2=None):
        super().__init__()
        assert(mode in ['training', 'testing', 'valid', 'real_testing'])
        self.mode = mode if mode=='real_testing' else 'training'
        self._mode = mode # store original mode 
        self.image_dir = image_dir
        self.data_list = json.load(open(data_path))
        
        if image_dir2 is not None: self.image_dir2 = image_dir2 
        if data_path2 is not None: self.data_list2 = json.load(open(data_path2))
        
        self.label2idx = {
            'Mitosis': 1, 'Non-mitosis': 0
        }
        
        if self._mode=='training':
            self.transforms = transforms.get_train_transforms(224)
        else:
            self.transforms = transforms.get_valid_transforms(224)
            
    def origin_crop(self, img, extra_size = 120):
        w, h = img.size
        x1 = y1 = extra_size
        x2 = w - extra_size
        y2 = h - extra_size
        return img.crop([x1, y1, x2, y2])
        
    def __len__(self):
        if hasattr(self, 'data_list2'):
            return len(self.data_list) + len(self.data_list2)
        else:
            return len(self.data_list)
    
    def __getitem__(self, idx):
        
        chosen_data_list = self.data_list
        chosen_image_dir = self.image_dir
        if idx >= len(self.data_list):
            idx = idx - len(self.data_list)
            chosen_data_list = self.data_list2
            chosen_image_dir = self.image_dir2
        
        data = chosen_data_list[idx]
        data['chosen_image_dir'] = chosen_image_dir
        data['chosen_data_list'] = chosen_data_list
        try:
            name_mode = self.mode.split('_')[1] if self.mode=='real_testing' else self.mode
            img_path = os.path.join(chosen_image_dir, f"{name_mode}_{data['id']}.jpg")
            img = Image.open(img_path).convert("RGB")
        except:
            for name_mode in ['testing', 'real_testing']:
                img_path = os.path.join(chosen_image_dir, f"{name_mode}_{data['id']}.jpg")
                if os.path.isfile(img_path): 
                    img = Image.open(img_path).convert("RGB")
                    
        
        # img = Image.open(img_path).convert("RGB")
        img = img.filter(ImageFilter.SHARPEN)
        img2 = self.origin_crop(img)
        
        x1 = self.transforms(image=np.array(img))['image']
        x2 = self.transforms(image=np.array(img2))['image']
        x = torch.cat([x1, x2], dim=0)  # 2CxHxW
        if self.mode=='real_testing':
            return x, data
        
        cls_num = self.label2idx[data['label']]
        
        vector = [0, 0]
        vector[int(cls_num)] = 1.0 if 'prob' not in data else float(data['prob'])
        # vector[int(cls_num)] = 1.0 if vector[int(cls_num)] >= 0.95 else vector[int(cls_num)]
        vector[int(1-cls_num)] = 1 - vector[int(cls_num)]
        
        y = torch.tensor(vector).float()
        
        return x, y, data
        
        