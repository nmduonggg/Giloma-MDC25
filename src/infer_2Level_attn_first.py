import torch
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
import numpy as np

def collate_infer(batch):
    inputs, data = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    # data = torch.stack(data, dim=0)
    return inputs, data

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/OneShotTesting/real_testing'
test_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/OneShotTesting/real_testing_data.json'

# TEST_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/OneShotTesting/Reprocess_PublicTesting_to_Check/real_testing'
# test_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/OneShotTesting/Reprocess_PublicTesting_to_Check/real_testing_data.json'

# Training hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                        # Adjust based on your system

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory to save model checkpoints
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================ #
#        Data Preparation        #
# ============================ #

# Initialize datasets
test_dataset = Patch_2Level_Dataset_MitosisGen(image_dir=TEST_DIR, data_path=test_data_path, mode='real_testing')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_infer, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
model = UNI_2Level_attn_first(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
if hasattr(model, "apply_lora_to_vit"):
    model.apply_lora_to_vit(lora_r=16, lora_alpha=32, first_layer_start=10)
    model.enable_lora_training()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
        
# checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/_good_checkpoints/best_model_UNI_attn_cutmix_2test_100.pth'
checkpoint_path = checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/_good_checkpoints/best_model_UNI_attn_cutmix_2test_99,36.pth'
model.load_state_dict(torch.load(checkpoint_path))

model = model.to(DEVICE)

def get_percentile(datas, n):
    datas = np.array(datas)
    return np.percentile(datas, n)
    

def infer_model(model):
    model.eval()
    
    dataloader = test_loader
    all_preds, all_outputs = [], []
    
    submission = {
        'Row ID': [],
        'Image ID': [],
        'Label ID': [],
        'Prediction': []
    }
    
    # Iterate over data
    cnt = 1
    for batch in tqdm(dataloader, total=len(dataloader)):
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
            
        # if probs[0] < 0.8: continue
        
        all_preds.extend(preds.cpu().numpy())
        all_outputs.extend(probs.cpu().numpy())
            
        submission['Row ID'] += list(range(cnt, cnt + len(datas)))
        submission['Image ID'] += [
            os.path.basename(data['original_json_file'][:-5]) for data in datas]
        submission['Label ID'] += [
            data['label'] for data in datas]
        
        cnt += BATCH_SIZE
        
    submission['Prediction'] = all_preds
    
    low_conf_cnt = 0
    for idx, prob in enumerate(all_outputs):
        if prob < 0.95: 
            print(idx)
            low_conf_cnt += 1
    print("Num of low conf: ", low_conf_cnt)
    
    for k, v in submission.items():
        print(k, len(v))
        
    tau = get_percentile(all_outputs, 3)
    print(tau)
        
    return submission

if __name__ == '__main__':
    # Start training
    login() # login to get foundation model
    submission = infer_model(model)
    df = pd.DataFrame.from_dict(submission)
    df.to_csv("submission.csv", index=False)