import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from huggingface_hub import login
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ResNetUNI_RealTime(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # login()
        
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True,
                                  num_classes=0)
        self.main_enc = enc
        self.context_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        self.main_enc = enc
        
        self.kd_projector = nn.Linear(2048, 1024)
        self.main_projector = nn.Linear(1024, 1024)
        
        self.classifier = nn.Linear(1024, num_classes)
        self.reweight = nn.Sequential(
            nn.Linear(1024*2, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.Sigmoid())
        self.tile_encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        
        self.tile_encoder.eval()    # tile for KD only
        
    def forward(self, x, context):
        
        with torch.no_grad():
            x_kd = self.tile_encoder(context)
            x_main = self.tile_encoder(x)
        x_cont = x_kd
        x_main = self.main_projector(x_main)
        
        x_fuse = x_cont
        for _ in range(3):
            weight = self.reweight(torch.concat([x_main, x_cont], dim=1))
            x_fuse = x_main * weight + x_cont * (1-weight)
            
        x = x_fuse
        
        # the below return is for the best
        return self.classifier(x), x_fuse, x_cont, x_kd
        
    
        