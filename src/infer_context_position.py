import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.PositionOriginalPatchDataset import OriginalPatchDataset
from data.DualPatchDataset import DualPatchDataset
# from model.resnet50 import ResNet50
from model.ResNet50_position import ResNet50_position
import os
import time
import pandas as pd
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

def collate_infer(batch):
    inputs, data, contexts, positions = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    contexts = torch.stack(contexts, dim=0)
    positions = torch.stack(positions, dim=0)
    return inputs, data, contexts, positions

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/real_testing'       # Replace with your training data path
test_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/real_testing_data.json'


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
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================ #
#        Data Preparation        #
# ============================ #

# Initialize datasets
# test_dataset = IndependentPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='testing')  # Add transforms if needed    # Add transforms if needed
# Initialize dataloaders
test_dataset = OriginalPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='real_testing')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_infer, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
model = ResNet50_position(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
# model = ResNetProvKD(num_classes=2)
# model = ProvGigaPath(num_classes=2)

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
        elif len(batch)==4:
            inputs, datas, context, position = batch
        inputs = inputs.to(DEVICE)
        context = context.to(DEVICE)
        position = position.to(DEVICE)

        submission['Row ID'] += list(range(cnt, cnt + len(datas)))
        submission['Image ID'] += [
            os.path.basename(data['original_json_file'][:-5]) for data in datas]
        submission['Label ID'] += [
            data['label'] for data in datas]

        with torch.no_grad():
            outputs = model(inputs, context, position)[0]
            outputs = F.softmax(outputs, dim=-1)
            probs, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
        cnt += BATCH_SIZE
    submission['Prediction'] = all_preds
    
    low_conf_cnt = 0
    for idx, prob in enumerate(all_outputs):
        if prob < 0.8: 
            print(idx)
            low_conf_cnt += 1
    print("Num of low conf: ", low_conf_cnt)
    
    for k, v in submission.items():
        print(k, len(v))
        
    return submission

if __name__ == '__main__':
    # Start training
    checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_ResNet_Context_Position_122824only.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    submission = infer_model(model)
    df = pd.DataFrame.from_dict(submission)
    df.to_csv("submission.csv", index=False)
