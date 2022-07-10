import torch
import torch.nn.functional as F
import os

def cal_loss(pred, gt, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gt = gt.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gt, reduction='mean')

    return loss

class IOStream():
    def __init__(self, file_name):
        self.f = os.path.join(file_name+'.log')

    def cprint(self, text):
        print(text)
        with open(self.f, 'a') as f:
            f.write(text+'\n')
            f.flush()