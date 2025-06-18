import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
from data import transforms
from labelme import utils as lbl_utils

class DualPatchDataset(Dataset):
    def __init__(self, image_dir, data_path, mode):
        super().__init__()
        assert(mode in ['training', 'testing', 'valid', 'real_testing'])
        self.mode = mode if mode=='real_testing' else 'training'
        self._mode = mode # store original mode 
        self.image_dir = image_dir
        self.data_list = json.load(open(data_path))
        
        self.label2idx = {
            'Mitosis': 1, 'Non-mitosis': 0
        }
        
        if mode=='training': 
            self.cell_transforms = transforms.CELL_TRANSFORMS
            self.roi_transforms = transforms.ROI_TRANSFORMS
        else: 
            self.roi_transforms = transforms.ROI_TRANSFORMS
            self.cell_transforms = transforms.TEST_TRANSFORMS
        
    def __len__(self):
        return len(self.data_list) * 2 if 'test' not in self._mode else len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        
        ## get original large slide
        original_json_file = data['original_json_file']
        original_data = json.load(open(original_json_file))
        ori_img = lbl_utils.img_b64_to_arr(original_data.get("imageData"))
        ori_img = Image.fromarray(ori_img).convert("RGB")
        
        try:
            img_path = os.path.join(self.image_dir, f"{self.mode}_{data['id']}.jpg")
            img = Image.open(img_path).convert("RGB")
        except:
            img_path = os.path.join(self.image_dir, f"testing_{data['id']}.jpg")
            img = Image.open(img_path).convert("RGB")
        
        if idx >= len(self.data_list):
            img = transforms.horizontal_flip(img)
            ori_img = transforms.horizontal_flip(img)
        
        x = self.cell_transforms(img)
        ox = self.roi_transforms(ori_img)
        
        if self.mode=='real_testing':
            return x, data, ox
        
        cls_num = self.label2idx[data['label']]
        y = torch.tensor(cls_num).long()
        # print(y)
        return x, y, ox
        
        