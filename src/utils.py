import os
from PIL import Image
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from icecream import ic
from data.transforms import get_train_transforms_strong

def hard_voting(models, inputs, context, device='cpu'):
    votes = []
    for model in models:
        model.to(device)
        with torch.no_grad():
            outputs = model(inputs, context)[0]
        _, predictions = torch.max(outputs, dim=1)
        votes.append(predictions)
        model.to('cpu') # save mem
        
    # Stack predictions and compute mode for majority vote
    votes = torch.stack(votes, dim=0)  # Shape: (num_models, batch_size)
    
    if (votes - torch.max(votes, dim=0)[1]).sum() > 0:
        print(votes.tolist())
    
    majority_vote, _ = torch.mode(votes, dim=0)
    return majority_vote

def cutmix(batch, alpha):
    data, targets, strong_data = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_strong_data = strong_data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    strong_data[:, :, y0:y1, x0:x1] = shuffled_strong_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets, strong_data

def origin_crop(img, extra_size = 120):
    w, h = img.size
    x1 = y1 = extra_size
    x2 = w - extra_size
    y2 = h - extra_size
    return img.crop([x1, y1, x2, y2])

class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha
        self.img_transform = get_train_transforms_strong(224)

    def __call__(self, batch):
        # batch = torch.utils.data.dataloader.default_collate(batch)
        images, targets, datas = zip(*batch)
        images = torch.stack(images, dim=0)
        original_targets = torch.stack(targets, dim=0)
        strong_images = list()
        
        for data in datas:
            chosen_image_dir = data['chosen_image_dir']
            for name_mode in ['training', 'testing', 'real_testing']:
                img_path = os.path.join(chosen_image_dir, f"{name_mode}_{data['id']}.jpg")
                if os.path.isfile(img_path): 
                    img = Image.open(img_path).convert("RGB")
            
            img2 = origin_crop(img, extra_size=120)
            x1 = self.img_transform(image=np.array(img))['image']
            x2 = self.img_transform(image=np.array(img2))['image']
            x = torch.cat([x1, x2], dim=0)
            strong_images.append(x)
            
        strong_images = torch.stack(strong_images, dim=0)
        
        batch = (images, original_targets, strong_images)
        data, targets, strong_data = cutmix(batch, self.alpha)
        return data, targets, strong_data, original_targets

class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        targets1 = targets1.to(preds.device)
        targets2 = targets2.to(preds.device)
        
        # loss = lam * self.kl_div(
        #     F.log_softmax(preds, dim=1), targets1) + (1 - lam) * self.kl_div(
        #         F.log_softmax(preds, dim=1), targets2)
        
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
        # return loss