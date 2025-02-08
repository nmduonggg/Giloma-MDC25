import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from huggingface_hub import login
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ResNet50_position(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        login()
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True,
                                  num_classes=0)
        self.main_enc = enc
        self.context_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        self.position_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        
        self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.kd_projectors = nn.ModuleList(
            [nn.Linear(2048, 1536) for _ in range(3)])
        
        
        self.classifier = nn.Linear(1536*2, num_classes)
    
        
    def forward(self, x, context, position):
        x_main = self.kd_projectors[0](self.main_enc(x))
        x_cont = self.kd_projectors[1]((self.context_processor(context)))
        
        alpha_pos = torch.sigmoid(
            self.kd_projectors[2](self.position_processor(position)))
        
        x = alpha_pos * x_main + (1-alpha_pos) * x_cont
        x_out = torch.cat([x, x_main], dim=-1)
        
        x_kd = None
        if self.train:
            with torch.no_grad():
                x_kd = self.tile_encoder(context)
        out = self.classifier(x_out)
        return out, x_main, x_cont, x_kd 
    
        