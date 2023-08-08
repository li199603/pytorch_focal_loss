import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # [C,]

    def forward(self, pred_logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # pred_logit [B, C]  or  [B, C, X1, X2, ...]
        # label [B, ]  or  [B, X1, X2, ...]
        B, C = pred_logit.shape[:2]  # batch size and number of categories
        if pred_logit.dim() > 2:
            # e.g. pred_logit.shape is [B, C, X1, X2]   
            pred_logit = pred_logit.reshape(B, C, -1)  # [B, C, H, W] => [B, C, H*W]
            pred_logit = pred_logit.transpose(1, 2)    # [B, C, H*W] => [B, H*W, C]
            pred_logit = pred_logit.reshape(-1, C)   # [B, H*W, C] => [B*H*W, C]   set N = B*H*W
        label = label.reshape(-1)  # [N, ]

        logpt = torch.log_softmax(pred_logit, dim=-1)  # [N, C]
        logpt = logpt.gather(1, label[:, None]).squeeze()  # [N,]
        pt = torch.exp(logpt)  # [N,]
        
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = torch.ones((C,), dtype=torch.float, device=pred_logit.device)
        alpha = alpha.gather(0, label)  # [N,]
        
        loss = -1 * alpha * torch.pow(1 - pt, self.gamma) * logpt
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
    
    focal_loss = FocalLoss(gamma=0.0, alpha=alpha)
    loss1 = focal_loss(pred_logit1, label)
    loss1.backward()
    
    loss2 = F.cross_entropy(pred_logit2, label, weight=alpha)
    loss2.backward()
    
    print(loss1)
    print(loss2)
    print(pred_logit1.grad[1, 2, 3, 4])
    print(pred_logit2.grad[1, 2, 3, 4])
    
    
    