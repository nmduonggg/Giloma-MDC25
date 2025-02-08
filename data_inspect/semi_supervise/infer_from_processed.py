import sys
sys.path.append("/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.DualPatchDataset import DualPatchDataset
# from model.resnet50 import ResNet50
from model.ResNet50_ProvKD_wOriginal import ResNetProvKD
import os
import time
import pandas as pd
import copy
import json
from tqdm import tqdm

def collate_infer(batch):
    inputs, data, labels = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    return inputs, data, labels

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/real_testing'       # Replace with your training data path
test_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/real_testing_data.json'


# Training hyperparameters
BATCH_SIZE = 16
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

# ============================ #
#        Data Preparation        #
# ============================ #

# Initialize datasets
# test_dataset = IndependentPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='testing')  # Add transforms if needed    # Add transforms if needed
# Initialize dataloaders
test_dataset = DualPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='real_testing')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_infer, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
# model = ResNet50(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
model = ResNetProvKD(num_classes=2)
# model = ProvGigaPath(num_classes=2)

index2class = ['Non-mitosis', 'Mitosis']

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
    all_datas = list()
    for batch in tqdm(dataloader, total=len(dataloader)):
        if len(batch)==2:
            inputs, datas = batch
        elif len(batch)==3:
            inputs, datas, context = batch
        inputs = inputs.to(DEVICE)
        context = context.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs, context)[0]
            outputs = F.softmax(outputs, dim=-1)
            probs, preds = torch.max(outputs, 1)
            
            preds = preds.cpu().numpy().tolist()
            probs = probs.cpu().numpy().tolist()
            for prob, pred, data in zip(probs, preds, datas):
                if prob < 0.9: continue
                data['label'] = index2class[int(pred)]
                all_datas.append(data)
    
    print(f"Total: {len(all_datas)} confident predictions")
        
    return all_datas

if __name__ == '__main__':
    # Start training
    checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/good_checkpoints/best_model_ResNet_Prov_dual_98,11.pth'
    final_external_json_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/all_data.json'
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    all_datas = infer_model(model)
    with open(final_external_json_path, 'w') as f:
        json.dump(all_datas, f, indent=4)
