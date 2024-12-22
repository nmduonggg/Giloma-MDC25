import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from huggingface_hub import login
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ProvGigaPath(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        login()
        
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True,
                                  num_classes=0)
        self.context_processor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.main_enc = enc
        self.ln = nn.Linear(2048, 1024)
        self.classifier = nn.Linear(2560, num_classes)
        
        self.context_processor.eval()
        
    def forward(self, x, context):
        x_main = self.ln(self.main_enc(x))
        with torch.no_grad():
            x_cont = self.context_processor(context)
        
        x = torch.concat([x_main, x_cont], dim=1)
        return self.classifier(x), x_main
    
        