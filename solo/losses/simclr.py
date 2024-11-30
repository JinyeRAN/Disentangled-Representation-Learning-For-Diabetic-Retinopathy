import torch
import torch.nn.functional as F

def simclr_loss_func(z, indexes, temperature: float = 0.2):
    z = F.normalize(z, dim=-1)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / temperature)

    indexes = indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == indexes
    pos_mask.fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss
