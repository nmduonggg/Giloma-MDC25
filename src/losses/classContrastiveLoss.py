import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        norm_embeddings = F.normalize(embeddings, dim=1)

        similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)

        # Create positive and negative masks
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        negative_mask = (1 - positive_mask).float()

        # Apply temperature
        sim_matrix_temp = similarity_matrix / self.temperature

        # Compute numerator and denominator in a vectorized manner
        exp_sim = torch.exp(sim_matrix_temp)
        numerator = exp_sim * positive_mask
        denominator = exp_sim.sum(dim=1, keepdim=True)

        # Avoid division by zero
        numerator = numerator.clamp(min=1e-9)
        loss = -torch.log(numerator / denominator)

        if loss.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        return loss.mean()