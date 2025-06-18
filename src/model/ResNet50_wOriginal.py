import torch
import timm 
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True,
                                  num_classes=0)
        self.context_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        self.main_enc = enc
        self.classifier = nn.Linear(2048*2, num_classes)
        
    def forward(self, x):
        context = x[:, :3, ...]
        x = x[:, 3:, ...]
        x_main = self.main_enc(x)
        x_cont = self.context_processor(context)
        
        x = torch.concat([x_main, x_cont], dim=1)
        return self.classifier(x), x_main
        