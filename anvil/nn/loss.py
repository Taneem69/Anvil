"""Loss functions as Modules (this wrappers over nn.functional)"""

from . import functional as F
from .module import Module

class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)



class MSELoss(Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)