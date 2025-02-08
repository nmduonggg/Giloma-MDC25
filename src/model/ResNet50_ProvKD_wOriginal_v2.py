import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from huggingface_hub import login
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ResNetProvKD_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        login()
        
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True,
                                  num_classes=0)
        self.main_enc = enc
        self.context_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        self.main_enc = enc
        
        
        self.kd_projector = nn.Linear(2048, 1536)
        self.classifier = nn.Linear(2048 + 1536, num_classes)
        
        self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        
        self.tile_encoder.eval()    # tile for KD only
        
    def forward(self, x, context):
        x_main = self.main_enc(x)
        x_cont = self.context_processor(context)
        
        x_kd = None
        if self.train:
            with torch.no_grad():
                x_kd = self.tile_encoder(context)
        x_cont = self.kd_projector(x_cont)
        x = torch.concat([x_main, x_cont], dim=1)
        return self.classifier(x), x_main, x_cont, x_kd
    
        