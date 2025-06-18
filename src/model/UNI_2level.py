import torch
import torch.nn as nn
import timm
import loralib as lora


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityContrastiveLoss(nn.Module):
    """
    Similarity Contrastive Loss:
    - Encourages positive pairs (same index in batch) to have high similarity.
    - Encourages negative pairs (different indices) to have low similarity.
    """

    def __init__(self, margin=0.5):
        """
        :param margin: Margin for negative pairs (default: 0.5)
        """
        super(SimilarityContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2):
        """
        Compute Similarity Contrastive Loss.

        :param x1: Tensor of shape (B, D) - First batch of vectors
        :param x2: Tensor of shape (B, D) - Second batch of vectors
        :return: Similarity contrastive loss
        """
        B, D = x1.shape

        # Compute cosine similarity matrix between all pairs
        cos_sim_matrix = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)  # (B, B)

        # Create labels: Positive pairs are diagonal (same index), others are negatives
        labels = torch.eye(B, device=x1.device)  # Identity matrix: 1 on diagonal, 0 elsewhere

        # Loss for positive pairs (diagonal) -> encourage similarity close to 1
        positive_loss = (1 - cos_sim_matrix) * labels  # Only for positive pairs

        # Loss for negative pairs (off-diagonal) -> encourage similarity < margin
        negative_loss = torch.clamp(cos_sim_matrix - self.margin, min=0) * (1 - labels)

        # Compute total loss (average over all pairs)
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= B  # Normalize by batch size

        return loss


class UNI_2Level(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.simCL = SimilarityContrastiveLoss(0.2)
        # login()
        self.enc1 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.enc2 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.classifier1 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes))
        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes))
        
        self.fusion = nn.Linear(1024 * 2, 1024)
        
    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]
        
        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        
        x = self.fusion(torch.cat([x1, x2], dim=1))
        # out1 = self.classifier1(x1)
        
        # x = self.fusion(torch.cat([x1, x2], dim=1))
        sim_loss = self.simCL(x1, x2)
        out1 = self.classifier1(x)
        # out2 = self.classifier1(x2)
        
        # out = (out1 + out2) * 0.5
        
        # the below return is for the best
        return out1, 0, 0
    
    def apply_lora_to_vit(self, lora_r, lora_alpha, first_layer_start=15):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        for enc in [self.enc1, self.enc2]:
            # Step 1: Collect the names of layers to replace
            layers_to_replace = []
            
            for name, module in enc.named_modules():
                if isinstance(module, nn.Linear) :
                    if ('qkv' in name or 'proj' in name) and (int(name.split('.')[1]) >= first_layer_start):
                        # Collect layers for replacement (store name and module)
                        layers_to_replace.append((name, module))
            
            # Step 2: Replace the layers outside of the iteration
            for name, module in layers_to_replace:
                # Create the LoRA-augmented layer
                lora_layer = lora.Linear(module.in_features, module.out_features, r=lora_r, lora_alpha=lora_alpha)
                # Copy weights and bias
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias.data = module.bias.data.clone()

                # Replace the layer in the model
                parent_name, layer_name = name.rsplit('.', 1)
                parent_module = dict(enc.named_modules())[parent_name]
                setattr(parent_module, layer_name, lora_layer)

    # Additional helper to enable LoRA fine-tuning
    def enable_lora_training(self):
        # Set LoRA layers to be trainable, freeze others
        for param in self.enc1.parameters():
            param.requires_grad = False
        for name, param in self.enc1.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                
        for param in self.enc2.parameters():
            param.requires_grad = False
        for name, param in self.enc2.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Enable gradients for the classifier head
        for param in self.classifier1.parameters():
            param.requires_grad = True
        for param in self.classifier2.parameters():
            param.requires_grad = True

        
    
        