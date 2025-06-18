import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data.Patch_2Level_Dataset import Patch_2Level_Dataset
from data.Patch_2Level_Dataset_MitosisGen import Patch_2Level_Dataset_MitosisGen
from data.Patch_2Level_Dataset_BiGen import Patch_2Level_Dataset_BiGen
from model.UNI_2level_attn_first import UNI_2Level_attn_first
import os
import json
from tqdm import tqdm
from huggingface_hub import login
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from icecream import ic

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TEST_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/training'
test_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches_remove_test/valid_data.json'
GEN_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/real_testing'
gen_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/ovelapped_predicted_real_testing_data_99,36_and-100.json'
GEN_DIR1 = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/real_testing'
gen_data_path1 = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/predicted_real_testing_data_100.json'

mode = 'valid'
WRONG_DIR = './wrong_eval'
os.makedirs(WRONG_DIR, exist_ok=True)

def save_im(im_id, mode):
    im = plt.imread(os.path.join(TEST_DIR, f'training_{im_id}.jpg'))
    plt.imsave(os.path.join(WRONG_DIR, f'training_{im_id}.jpg'), im)
    
    return

def collate_fn(batch):
    images, targets, datas = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, datas

# Training hyperparameters
BATCH_SIZE = 64
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

# Initialize dataloaders
test_dataset1 = Patch_2Level_Dataset_MitosisGen(image_dir=TEST_DIR, data_path=test_data_path, mode='valid')
test_dataset2 = Patch_2Level_Dataset_MitosisGen(image_dir=GEN_DIR, data_path=gen_data_path, mode='valid') 
test_dataset3 = Patch_2Level_Dataset_MitosisGen(image_dir=GEN_DIR1, data_path=gen_data_path1, mode='valid') 
# Initialize dataloaders
test_loader1 = DataLoader(test_dataset1, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
test_loader3 = DataLoader(test_dataset3, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# Initialize the model

# Initialize the model
model = UNI_2Level_attn_first(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
if hasattr(model, "apply_lora_to_vit"):
    model.apply_lora_to_vit(lora_r=16, lora_alpha=32, first_layer_start=10)
    model.enable_lora_training()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()

checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/checkpoints/best_model_UNI_attn_cutmix_2test.pth'
model.load_state_dict(torch.load(checkpoint_path))
        
model = model.to(DEVICE)
# model = ProvGigaPath(num_classes=2)

def get_diff_indices(arr1, arr2):
    return np.where(np.array(arr1) != np.array(arr2))[0]

data_list = json.load(open(test_data_path))

def infer_model(model):
    model.eval()
    
    for dataloader in [test_loader1, test_loader2, test_loader3]:
        all_preds, all_labels = [], []
        all_outputs = []
        
        output_per_class = {
            0: [], 1: []
        }
        
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
                inputs, labels, datas = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.no_grad():
                outputs = model(inputs)[0]
                outputs = F.softmax(outputs, dim=-1)
                probs, preds = torch.max(outputs, 1)
                all_outputs.extend(probs.cpu().numpy().tolist())
                
            probs = probs.cpu().numpy().tolist()
            preds = preds.cpu().numpy().tolist()
            labels = torch.argmax(labels, dim=1).cpu().numpy().tolist()
            for ip, pred in enumerate(preds):
                if pred == labels[ip]:
                    output_per_class[int(pred)] += [probs[int(ip)]]
                
            all_labels.extend(labels)
            all_preds.extend(preds)
            cnt += 1
        # submission['Prediction'] = all_preds
        
        low_conf_cnt = 0
        for idx, prob in enumerate(all_outputs):
            if prob < 0.95: 
                # print(idx)
                low_conf_cnt += 1
        print("Num of low conf: ", low_conf_cnt)
        
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # Choose 'macro', 'micro', or 'binary' as needed
        acc = (np.array(all_labels) == np.array(all_preds)).mean()
        
        diff_indices = get_diff_indices(all_preds, all_labels).tolist()
        for idx in diff_indices:
            ic(all_labels[idx], all_outputs[idx])
            
            print('-------')
        
        print(f"F1: {epoch_f1} - Acc: {acc}")
        
        

if __name__ == '__main__':
    # Start training
    login('')
    infer_model(model)
