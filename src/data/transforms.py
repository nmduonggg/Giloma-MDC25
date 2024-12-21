from torchvision import transforms

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),                     # Resize the image to 256x256 pixels
    transforms.RandomCrop(224),                         # Randomly crop a 224x224 region
    transforms.RandomHorizontalFlip(),                  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),
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
