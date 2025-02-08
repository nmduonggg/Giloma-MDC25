import torch

def hard_voting(models, inputs, context, device='cpu'):
    votes = []
    for model in models:
        model.to(device)
        with torch.no_grad():
            outputs = model(inputs, context)[0]
        _, predictions = torch.max(outputs, dim=1)
        votes.append(predictions)
        model.to('cpu') # save mem
        
    # Stack predictions and compute mode for majority vote
    votes = torch.stack(votes, dim=0)  # Shape: (num_models, batch_size)
    
    if (votes - torch.max(votes, dim=0)[1]).sum() > 0:
        print(votes.tolist())
    
    majority_vote, _ = torch.mode(votes, dim=0)
    return majority_vote