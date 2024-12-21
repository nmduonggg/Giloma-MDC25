import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.IndependentPatchDataset import IndependentPatchDataset
from model.resnet50 import ResNet50
import os
import time
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

# ============================ #
#        Configuration          #
# ============================ #

# Paths to your data directories
TRAIN_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/training'       # Replace with your training data path
VAL_DIR = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/training'           # Replace with your validation data path
train_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/training_data.json'
valid_data_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/valid_data.json'

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
train_dataset = IndependentPatchDataset(image_dir=TRAIN_DIR, data_path=train_data_path, mode='training')  # Add transforms if needed
val_dataset = IndependentPatchDataset(image_dir=VAL_DIR, data_path=valid_data_path, mode='valid')      # Add transforms if needed

# Initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Dictionary to hold dataset sizes
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# ============================ #
#        Model Initialization    #
# ============================ #

# Initialize the model
model = ResNet50(num_classes=2)      # Set pretrained=True if you want to use pretrained weights

# Move the model to the configured device
model = model.to(DEVICE)

# ============================ #
#     Loss Function & Optimizer  #
# ============================ #

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Optionally, define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============================ #
#          Training Loop          #
# ============================ #

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

    best_model_wts = copy.deepcopy(model.state_dict())
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
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            all_preds, all_labels = [], []

            # Iterate over data
            for inputs, labels in tqdm(dataloader, total=len(dataloader)):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # Choose 'macro', 'micro', or 'binary' as needed

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            # Deep copy the model if it has better accuracy
            # if phase == 'val' and epoch_acc > best_acc:\
            if phase=='val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model weights
                torch.save(best_model_wts, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
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

    # Save the final trained model
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
