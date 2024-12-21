import torch
from torch import nn
import torch.nn.functional as F

class ClassContrastiveLoss(nn.Module):
    ## consider binary only
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, labels):
        bs, dim = preds.shape
        norm_preds = F.normalize(preds, dim=-1)
        sim_mat = norm_preds @ norm_preds.T
        sim_mat = F.sigmoid(sim_mat)
        label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()
        
        # Compute cross-entropy loss element-wise between similarity_matrix and label_matrix
        loss = - (label_mat * torch.log(sim_mat) + (1 - label_mat) * torch.log(1 - sim_mat))
        
        return loss.mean()