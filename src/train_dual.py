import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.DualPatchDataset import DualPatchDataset
from data.RichDualPatchDataset import RichDualPatchDataset
from model.ResNet50_ProvKD_wOriginal import ResNetProvKD
from model.ResNet50_ProvKD_wOriginal_v2 import ResNetProvKD_v2
from model.ResNet50_wOriginal import ResNet50
from losses.classContrastiveLoss import ClassContrastiveLoss
from losses.FocalLoss import FocalLoss
import os
import time
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TRAIN_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/real_testing'
VAL_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/real_testing'
train_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/training_data.json'
valid_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/valid_data.json'

# TRAIN_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/training'       # Replace with your training data path
# VAL_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/training'           # Replace with your validation data path
# train_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/training_data.json'
# valid_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches_122824/valid_data.json'

# Training hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 32                        # Adjust based on your system

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
train_dataset = RichDualPatchDataset(image_dir=TRAIN_DIR, data_path=train_data_path, mode='training')  # Add transforms if needed
val_dataset = RichDualPatchDataset(image_dir=VAL_DIR, data_path=valid_data_path, mode='valid')      # Add transforms if needed

# Initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

# Dictionary to hold dataset sizes
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# ============================ #
#        Model Initialization    #
# ============================ #

# Initialize the model
# model = ResNet50(num_classes=2)      # Set pretrained=True if you want to use pretrained weights
# model = ResNetProvKD_v2(num_classes=2)
model = ResNetProvKD(num_classes=2)
# checkpoint_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/src/checkpoints/best_model_ResNet_ProvKD_Dual_pretrain_augment_wO.pth'
# model.load_state_dict(torch.load(checkpoint_path))

# Move the model to the configured device
model = model.to(DEVICE)

# ============================ #
#     Loss Function & Optimizer  #
# ============================ #

# Define the loss function
criterion = nn.CrossEntropyLoss()
focal_criterion = FocalLoss(gamma=2.0)
ctrs_loss = ClassContrastiveLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optionally, define a learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-8)

def freeze_modules(model):
    # blacklist = ['classifier', 'main_enc']
    # for n, module in model.named_modules():
    #     if all(item not in n for item in blacklist) or isinstance(module, nn.BatchNorm2d):
    #         module.eval()
    #     else:
    #         # print(n)
    #         pass
    pass

# ============================ #
#          Training Loop          #
# ============================ #

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, name='model'):
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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    best_wrong_probs = 1e6

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                freeze_modules(model)
            else:
                model.eval()   # Set model to evaluation mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            all_wrong_probs = 0.0
            
            all_preds, all_labels = [], []

            # Iterate over data
            for inputs, labels, context in tqdm(dataloader, total=len(dataloader)):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                context = context.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track gradients only in train phase
                if phase == 'train':
                    
                    kd_feat, cont_feat = None, None
                    
                    try:
                        outputs, features = model(inputs, context)
                    except: 
                        outputs, features, cont_feat, kd_feat = model(inputs, context)
                    _, preds = torch.max(outputs, 1)
                    
                    wrong_instances = 1 - (preds == labels.data).int()
                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    # Create a mask for wrong predictions
                    wrong_instances = (preds != labels).float()

                    # Create a mask that keeps the values at the indices in `preds` only for the wrong instances
                    mask = torch.zeros_like(probs)  # Start with a tensor of zeros with the same shape as `probs`
                    mask[torch.arange(probs.size(0)), preds] = wrong_instances  # Set the mask to 1 at the wrong predicted indices

                    # Now, you can multiply `mask` with `probs` to get only the probabilities at the wrong predicted indices
                    wrong_probs = torch.sum(probs * mask) / torch.sum(mask)
                    
                    loss = criterion(outputs, labels) + 0.2 * focal_criterion(torch.softmax(outputs, dim=-1), labels)
                    
                    contrastive_loss = ctrs_loss(features, labels)
                    loss = loss + 0.1 * contrastive_loss
                    
                    loss = loss + wrong_probs
                    
                    if kd_feat is not None and cont_feat is not None:
                        kd_loss = torch.mean((cont_feat - kd_feat).pow(2))
                        loss = loss + 0.5 * kd_loss

                    # Backward pass and optimization only in train phase
                    loss.backward()
                    optimizer.step()
                
                elif phase=='val':
                    with torch.no_grad():
                        outputs = model(inputs, context)[0]
                        probs, preds = torch.max(outputs, 1)
                        wrong_instances = 1 - (preds == labels.data).int()
                        wrong_probs = torch.sum(probs * wrong_instances) / torch.sum(wrong_instances)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_wrong_probs += wrong_probs.detach().cpu().numpy()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # Choose 'macro', 'micro', or 'binary' as needed
            
            # Step the scheduler only in the train phase
            if phase == 'val':
                # scheduler.step(epoch_f1)
                scheduler.step()

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} WrongPrb: {all_wrong_probs:.4f}')

            # Deep copy the model if it has better accuracy
            # if phase == 'val' and epoch_acc > best_acc:\
            if phase=='val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'best_model_{name}_wO.pth'))
                    print(f"Best model updated at epoch {epoch + 1}")
                elif abs(epoch_f1 - best_f1) < 1e-6:
                    if best_wrong_probs > all_wrong_probs:
                        best_wrong_probs = all_wrong_probs
                        # Save the best model weights
                        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'best_model_{name}_wO.pth'))
                        print(f"Best model updated at epoch {epoch + 1}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation F1: {best_f1:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    
    name = 'Resnet_ProvKD_Dual_pretrain_augment'
    # name = 'ResNet_ProvKD_Dual_pretrain_augment'
    # Start training
    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, name=name)

    # Save the final trained model
    # final_model_path = os.path.join(CHECKPOINT_DIR, f'final_model_{name}_wO.pth')
    # torch.save(trained_model.state_dict(), final_model_path)
    # print(f"Final model saved to {final_model_path}")
