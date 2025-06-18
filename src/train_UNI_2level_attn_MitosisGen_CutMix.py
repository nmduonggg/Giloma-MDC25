import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.Patch_2Level_Dataset_MitosisGen import Patch_2Level_Dataset_MitosisGen
from model.UNI_2level_attn_first import UNI_2Level_attn_first
from losses.classContrastiveLoss import ClassContrastiveLoss
from losses.FocalLoss import FocalLoss

from utils import CutMixCollator, CutMixCriterion

import os
import time
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

from huggingface_hub import login

def collate_fn(batch):
    images, targets, datas = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, datas

login('')

# Paths to your data directories
TRAIN_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/training'       # Replace with your training data path
VAL_DIR = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches/training'           # Replace with your validation data path
train_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches_remove_test/training_data.json'
valid_data_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/by_patches_remove_test/valid_data.json'
EXTRA_DIR_TRAIN= '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/processed_format/real_testing'
extra_train_path = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/processed_format/filtered_data.json'

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                        # Adjust based on your system
  
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory to save model checkpoints
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize datasets
train_dataset = Patch_2Level_Dataset_MitosisGen(image_dir=TRAIN_DIR, data_path=train_data_path, mode='training', image_dir2=EXTRA_DIR_TRAIN, data_path2=extra_train_path)  # Add transforms if needed

# val_dataset = Patch_2Level_Dataset_MitosisGen(image_dir=VAL_DIR, data_path=valid_data_path, image_dir2=GEN_DIR, data_path2=gen_data_path,  mode='valid')
val_dataset = Patch_2Level_Dataset_MitosisGen(image_dir=VAL_DIR, data_path=valid_data_path,  mode='valid')

alpha = 0.9 # loop 1 + 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CutMixCollator(alpha), num_workers=NUM_WORKERS)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# Initialize the model
model = UNI_2Level_attn_first(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
if hasattr(model, "apply_lora_to_vit"):
    model.apply_lora_to_vit(lora_r=16, lora_alpha=32, first_layer_start=10)
    model.enable_lora_training()

# checkpoint_path = '/home/nmduongg/Gilioma-ISBI25/works/Giloma-MDC25/src/checkpoints/best_model_UNI_attn_cutmix_2test.pth'
# model.load_state_dict(torch.load(checkpoint_path))
model = model.to(DEVICE)

# ============================ #
#     Loss Function & Optimizer  #
# ============================ #

# Define the loss function
criterion = CutMixCriterion(reduction='mean')
ce_criterion = nn.CrossEntropyLoss()
fc_loss = FocalLoss(gamma=2.0)
ctrs_loss = ClassContrastiveLoss()

# Define the optimizer
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-7)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains the model and evaluates it on the validation set.

    Args:
        model: The neural network model.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        num_epochs: Number of training epochs.

    Returns:
        model: The trained model with the best validation accuracy.
    """
    since = time.time()
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluation mode
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            all_preds, all_labels = [], []

            # Iterate over data
            for batch in tqdm(dataloader, total=len(dataloader)):
                if len(batch)==3:
                    inputs, labels, datas = batch
                elif len(batch)==2:
                    inputs, labels = batch
                elif len(batch)==4:
                    inputs, cutmix_labels, strong_inputs, labels = batch
                    strong_inputs = strong_inputs.to(DEVICE)
                labels = torch.argmax(labels, dim=-1)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs, features, sim_loss = model(inputs)
                    if len(batch) > 3:
                        outputs, features, sim_loss = model(strong_inputs)
                        loss = criterion(outputs, cutmix_labels)
                    else:
                        outputs, features, sim_loss = model(inputs)
                        loss = ce_criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    contrastive_loss = ctrs_loss(features, labels)
                    loss = loss + contrastive_loss * 0.2 + sim_loss * 0.2   # first loop
                    # loss = loss + contrastive_loss * 0.1 + sim_loss * 0.1   # second loop
                    
                    # Backward pass and optimization only in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Step the scheduler only in the train phase
            if phase == 'train':
                scheduler.step()
                pass

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # Choose 'macro', 'micro', or 'binary' as needed

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
            
            if phase=='val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model weights
                torch.save(best_model_wts, os.path.join(CHECKPOINT_DIR, 'best_model_UNI_attn_cutmix_2test.pth'))
                print(f"Best model updated at epoch {epoch + 1}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation F1: {best_f1:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Start training
    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)