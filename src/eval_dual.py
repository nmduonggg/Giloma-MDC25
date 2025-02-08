import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data.RealTimeCroppedDualPatchDataset import RealTimeCroppedDualPatchDataset
from data.DualPatchDataset import DualPatchDataset
from model.ResNet50_wOriginal import ResNet50
from model.ResNet50_ProvKD_wOriginal import ResNetProvKD
from model.ResNet50_ProvKD_wOriginal_v2 import ResNetProvKD_v2
import os
import json
from tqdm import tqdm
from huggingface_hub import login
from sklearn.metrics import f1_score

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/training'       # Replace with your training data path
test_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/testing_data.json'


# Training hyperparameters
BATCH_SIZE = 32
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
test_dataset = DualPatchDataset(image_dir=TEST_DIR, data_path=test_data_path, mode='testing')
# Initialize dataloaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

# Initialize the model
# model = ResNet50(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
# model = ResNetProvKD_v2(num_classes=2)
model = ResNetProvKD(num_classes=2)
# model = ProvGigaPath(num_classes=2)

def get_diff_indices(arr1, arr2):
    return np.where(np.array(arr1) != np.array(arr2))[0]


data_list = json.load(open(test_data_path))

def infer_model(model):
    model.eval()
    dataloader = test_loader
    all_preds, all_labels = [], []
    all_outputs = []
    
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
            outputs = model(inputs, context)[0]
            outputs = F.softmax(outputs, dim=-1)
            probs, preds = torch.max(outputs, 1)
            all_outputs.extend(probs.cpu().numpy().tolist())
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        cnt += 1
    # submission['Prediction'] = all_preds
    
    low_conf_cnt = 0
    for idx, prob in enumerate(all_outputs):
        if prob < 0.8: 
            print(idx)
            low_conf_cnt += 1
    print("Num of low conf: ", low_conf_cnt)
    
    diff_indices = get_diff_indices(all_preds, all_labels).tolist()
    for idx in diff_indices:
        if idx >= len(data_list):
            print("Flipped")
        print(idx)
        print(all_outputs[idx % len(data_list)])
        print(data_list[idx % len(data_list)]['id'])
        
        print('-------')
        
    
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # Choose 'macro', 'micro', or 'binary' as needed
    
    return epoch_f1

if __name__ == '__main__':
    # Start training
    login()
    checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/good_checkpoints/best_model_ResNet_Prov_dual_98,11.pth'
    # checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_Resnet_ProvKD_Dual_scratch_augment_realtime_crop_wO.pth'
    # checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/good_checkpoints/best_model_ResNet_Prov_dual_98,11.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    metric = infer_model(model)
    # df = pd.DataFrame.from_dict(submission)
    # df.to_csv("submission.csv", index=False)
    print("F1: ", metric)
