import sys
import copy
import numpy as np
sys.path.append("/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/")

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.Patch_2Level_Dataset_MitosisGen import Patch_2Level_Dataset_MitosisGen
from model.UNI_2level_attn_first import UNI_2Level_attn_first
import os
import pandas as pd
import copy
from tqdm import tqdm
from huggingface_hub import login
from sklearn.metrics import f1_score

def collate_infer(batch):
    inputs, data = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    # labels = torch.stack(labels, dim=0)
    return inputs, data

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/processed_format/real_testing'       # Replace with your training data path
test_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/processed_format/real_testing_data.json'


# Training hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 20                       # Adjust based on your system

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory to save model checkpoints
CHECKPOINT_DIR = 'checkpoints'

test_dataset = Patch_2Level_Dataset_MitosisGen(image_dir=TEST_DIR, data_path=test_data_path, mode='real_testing')

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_infer, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model

# Initialize the model
model = UNI_2Level_attn_first(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
if hasattr(model, "apply_lora_to_vit"):
    model.apply_lora_to_vit(lora_r=16, lora_alpha=32, first_layer_start=10)
    model.enable_lora_training()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
        
checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/_good_checkpoints/best_model_UNI_attn_cutmix_2test_100.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.to(DEVICE)

index2class = ['Non-mitosis', 'Mitosis']

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = copy.deepcopy(json.load(f))
    return data
    
def write_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
        
def get_bounding_box(points):
    points = np.array(points)
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))
    
    return [x1, y1, x2, y2]

def remove_circular_refs(ob, _seen=None):
    if _seen is None:
        _seen = set()
    if id(ob) in _seen:
        # circular reference, remove it.
        return None
    _seen.add(id(ob))
    res = ob
    if isinstance(ob, dict):
        res = {
            remove_circular_refs(k, _seen): remove_circular_refs(v, _seen)
            for k, v in ob.items()}
    elif isinstance(ob, (list, tuple, set, frozenset)):
        res = type(ob)(remove_circular_refs(v, _seen) for v in ob)
    # remove id again; only *nested* references count
    _seen.remove(id(ob))
    return res


def infer_model(model):
    model.eval()
    
    # Iterate over data
    all_datas = list()
    
    mitosis_cnt = 0
    nonmitosis_cnt = 0
    idx = 0
    
    first_round = len(test_loader) // 3 # 52
    first_round = 5
    
    for batch in tqdm(test_loader, total=len(test_loader)):  
        if len(batch)==2:
            inputs, datas = batch
        elif len(batch)==3:
            inputs, datas, context = batch
        inputs = inputs.to(DEVICE)
        # context = context.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)[0]
            
        outputs = F.softmax(outputs, dim=-1)
        probs, preds = torch.max(outputs, 1)
        
        preds = preds.cpu().numpy().tolist()
        probs = probs.cpu().numpy().tolist()
        for prob, pred, data in zip(probs, preds, datas):
            # if prob <= 0.85: continue
            # data['points'] = get_bounding_box(data['points'])
            
            data['chosen_data_list'] = None
            data['label'] = index2class[int(pred)]
            data['prob'] = float(prob)
            
            all_datas.append(data)
            
            if pred: mitosis_cnt += 1
            else: nonmitosis_cnt += 1
            
                
        # if idx % 5==0: print(len(all_datas))
        idx += 1
    
    print(f"Total: {len(all_datas)} confident predictions")
    print(f"Mitosis: {mitosis_cnt} - NonMitosis: {nonmitosis_cnt}")
        
    return all_datas

if __name__ == '__main__':
    # Start training
    # checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/checkpoints/best_model_UNI_attn_cutmix_2test.pth'
    final_external_json_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/processed_format/all_data_1.json'
    # model.load_state_dict(torch.load(checkpoint_path))
    all_datas = infer_model(model)
    # all_datas = remove_circular_refs(all_datas)
    
    with open(final_external_json_path, 'w') as f:
        json.dump(all_datas, f, indent=4)

