import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.RichDualPatchDataset import RichDualPatchDataset
from data.DualPatchDataset import DualPatchDataset
# from model.resnet50 import ResNet50
from model.ResNet50_wOriginal import ResNet50
from model.ProvGigaPath_wOriginal import ProvGigaPath
from model.ResNet50_ProvKD_wOriginal import ResNetProvKD
import os
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login

def collate_infer(batch):
    inputs, data, labels = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    return inputs, data, labels

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
test_dataset = RichDualPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='real_testing')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_infer, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
# model = ResNetProvKD(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
# model = ResNetProvKD(num_classes=2)
# model = ProvGigaPath(num_classes=2)

def infer_model(models):
    models = [model.eval() for model in models]
    
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
        context = context.to(DEVICE)

        submission['Row ID'] += list(range(cnt, cnt + len(datas)))
        submission['Image ID'] += [
            os.path.basename(data['original_json_file'][:-5]) for data in datas]
        submission['Label ID'] += [
            data['label'] for data in datas]

        voting_probs = 0.0
        for model in models:
            model.to(DEVICE)
            with torch.no_grad():
                outputs = model(inputs, context)[0]
                outputs = F.softmax(outputs, dim=-1).cpu()
                voting_probs += outputs
                
                model.to('cpu')
        voting_probs /= len(models)
        
        probs, preds = torch.max(voting_probs, 1)
        
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
    login()
    # checkpoint_path1 = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_ResNet_ProvKD_Dual_pretrain_augment_wO.pth'
    checkpoint_path2 = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_Resnet_ProvKD_Dual_scratch_augment_realtime_crop_wO.pth'
    checkpoint_path3 = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/good_checkpoints/best_model_ResNet_Prov_dual_98,11.pth'
    
    num_models = 2
    models = [ResNetProvKD(num_classes=2) for _ in range(num_models)]
    checkpoint_paths = [checkpoint_path2, checkpoint_path3]
    for model, checkpoint_path in zip(models, checkpoint_paths):
        model.load_state_dict(torch.load(checkpoint_path))
    submission = infer_model(models)
    df = pd.DataFrame.from_dict(submission)
    df.to_csv("submission.csv", index=False)
