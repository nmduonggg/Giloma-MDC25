import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
from data import transforms
import random
from labelme import utils as lbl_utils

class RealTimeCroppedDualPatchDataset(Dataset):
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
        
        if self._mode=='training':
            self.transforms = transforms.get_train_transforms(224)
        else:
            self.transforms = transforms.get_valid_transforms(224)
        
        # if mode=='training': 
        #     self.cell_transforms = transforms.CELL_TRANSFORMS
        #     self.roi_transforms = transforms.ROI_TRANSFORMS
        # else: 
        #     self.roi_transforms = transforms.ROI_TRANSFORMS
        #     self.cell_transforms = transforms.TEST_TRANSFORMS
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        
        ## get original large slide
        original_json_file = data['original_json_file']
        original_data = json.load(open(original_json_file))
        ori_img = lbl_utils.img_b64_to_arr(original_data.get("imageData"))
        # ori_img = Image.fromarray(ori_img).convert("RGB")
        
        offset = 10
        global_offset = 100
        
        offset = int(random.random() * 3) + 10 if self._mode=='training' else offset
        global_offset = int(random.random() * 20) + 90 if self._mode=='training' else global_offset
    
        independent_crops = transforms.shapes_to_independent_labels(ori_img, original_data['shapes'], offset, global_offset)
        
        if self.mode=='real_testing':
            choice = None
            for candidate in independent_crops:
                img, local_img, label, _ = candidate
                if label==data['label']: choice = candidate
        else:
            choice = random.choice(independent_crops) 
        
        img, local_img, label, _ = choice
        
        # img.save('./sample.jpg')
        
        # ori_img = np.asarray(ori_img)
        h, w = ori_img.shape[:2]
        img = img.resize((w, h))
        local_img = img.resize((w, h))
        img = np.asarray(img)
        local_img = np.asarray(local_img)
        x, ox, local_x = self.transforms(images=[img, ori_img, local_img])['images']

        if self.mode=='real_testing':
            return x, data, local_x
        
        if 'Blank' in label: 
            label = data['label']
        cls_num = self.label2idx[label]
        y = torch.tensor(cls_num).long()
        # print(y)
        return x, y, local_x
        
        