import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # [C,]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.gamma, self.alpha)

def focal_loss(pred_logit: torch.Tensor,
               label: torch.Tensor,
               gamma: float,
               alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    # pred_logit [B, C]  or  [B, C, X1, X2, ...]
    # label [B, ]  or  [B, X1, X2, ...]
    B, C = pred_logit.shape[:2]  # batch size and number of categories
    if pred_logit.dim() > 2:
        # e.g. pred_logit.shape is [B, C, X1, X2]   
        pred_logit = pred_logit.reshape(B, C, -1)  # [B, C, X1, X2] => [B, C, X1*X2]
        pred_logit = pred_logit.transpose(1, 2)    # [B, C, X1*X2] => [B, X1*X2, C]
        pred_logit = pred_logit.reshape(-1, C)   # [B, X1*X2, C] => [B*X1*X2, C]   set N = B*X1*X2
    label = label.reshape(-1)  # [N, ]

    log_p = torch.log_softmax(pred_logit, dim=-1)  # [N, C]
    log_p = log_p.gather(1, label[:, None]).squeeze()  # [N,]
    p = torch.exp(log_p)  # [N,]
    
    if alpha is None:
        alpha = torch.ones((C,), dtype=torch.float, device=pred_logit.device)
    alpha = alpha.gather(0, label)  # [N,]
    
    loss = -1 * alpha * torch.pow(1 - p, gamma) * log_p
    return loss.sum() / alpha.sum()


if __name__ == "__main__":
    import numpy as np
    
    B, C, X1, X2 = 32, 4, 100, 200
    pred_logit = np.random.randn(B, C, X1, X2)
    pred_logit1 = torch.tensor(pred_logit, dtype=torch.float, requires_grad=True)
    pred_logit2 = torch.tensor(pred_logit, dtype=torch.float, requires_grad=True)
    
    label = np.random.randint(0, C, size=(B, X1, X2))
    label = torch.tensor(label, dtype=torch.long)
    
    alpha = np.abs(np.random.randn(C))
    alpha = torch.tensor(alpha, dtype=torch.float)
    
    loss1 = FocalLoss(gamma=0.0, alpha=alpha)(pred_logit1, label)
    loss1.backward()
    
    loss2 = F.cross_entropy(pred_logit2, label, weight=alpha)
    loss2.backward()
    
    print(loss1)
    print(loss2)
    print(pred_logit1.grad[1, 2, 3, 4])
    print(pred_logit2.grad[1, 2, 3, 4])
    
    
    