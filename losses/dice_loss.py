import torch
from torch import nn

class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,input, target):
        '''
        input: [N, C]
        target: [N, ]
        '''
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss



