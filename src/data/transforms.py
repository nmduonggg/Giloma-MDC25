from torchvision import transforms

CELL_TRANSFORMS = transforms.Compose([
    transforms.Resize((240, 240)),                     # Resize the image to 256x256 pixels
    transforms.RandomCrop(224),                         # Randomly crop a 224x224 region
    transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

ROI_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),                     # Resize the image to 256x256 pixels
    transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),                      # Resize the image to 256x256 pixels                     # Crop the center 224x224 region
    transforms.ToTensor(),                               # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

## normal function
def horizontal_flip(img):
    flipper = transforms.RandomHorizontalFlip(p=1)
    return flipper(img)
