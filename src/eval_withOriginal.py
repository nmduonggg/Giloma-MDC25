import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.IndependentPatchDataset import IndependentPatchDataset
from data.OriginalPatchDataset import OriginalPatchDataset
# from model.resnet50 import ResNet50
from model.resnet50_wOriginal import ResNet50
from model.ProvGigaPath_wOriginal import ProvGigaPath
import os
import time
import pandas as pd
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/training'       # Replace with your training data path
test_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/testing_data.json'


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
test_dataset = OriginalPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='valid')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
# model = ResNet50(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
model = ProvGigaPath(num_classes=2)
def infer_model(model):
    model.eval()
    dataloader = test_loader
    all_preds, all_labels = [], []
    
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
            inputs, labels = batch
        elif len(batch)==3:
            inputs, labels, context = batch
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        context = context.to(DEVICE)

        # submission['Row ID'].append(cnt)
        # submission['Image ID'].append(
        #     os.path.basename(data['original_json_file'][0])[:-5])
        # submission['Label ID'].append(data['label'][0])

        with torch.no_grad():
            outputs, _ = model(inputs, context)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        cnt += 1
    # submission['Prediction'] = all_preds
    
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # Choose 'macro', 'micro', or 'binary' as needed
    
    return epoch_f1

if __name__ == '__main__':
    # Start training
    checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_titan_wO.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    metric = infer_model(model)
    # df = pd.DataFrame.from_dict(submission)
    # df.to_csv("submission.csv", index=False)
    print("F1: ", metric)
