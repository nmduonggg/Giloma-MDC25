import torch
import timm 
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model_name = 'resnet50'
        model = timm.create_model(model_name, pretrained=True,
                                  num_classes=num_classes)
        self.model = model
        
    def forward(self, x):
        return self.model(x)
        