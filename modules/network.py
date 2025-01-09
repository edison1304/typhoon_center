import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        
    def forward(self, pred, label):
        return torch.mean(torch.sqrt(torch.sum((pred - label) ** 2, dim=1)))


def get_loss_function(config):
    if config['loss'] == 'mae':
        return MAELoss()
    else:
        raise ValueError("Invalid loss type.")

#optimizer, config, len(train_loader)
def get_scheduler(config, optimizer, loader_length):

    if config['lr_policy'] == 'constant': scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif config['lr_policy'] == 'linear':# 사용하려면 config에 niter, niter_decay 추가해야함
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - int(config['niter'])) / float(int(config['niter_decay']) + 1)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif config['lr_policy'] == 'step': scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['epoch'], gamma=float(config['gamma']))
    elif config['lr_policy'] == 'plateau': scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config['lr_policy'] == 'cosine': scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['epoch']), eta_min=0)
    elif config['lr_policy'] == 'onecycle' : scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["learning_rate"] * 10, steps_per_epoch=loader_length, 
                                                                                             div_factor=2, final_div_factor=10000, epochs=config["epoch"], pct_start=0.1)
    elif config['lr_policy'] == 'warmup_cosine':
        total_steps = config["epoch"] * loader_length
        warmup_steps = int(0.1 * total_steps)
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        raise NotImplementedError(f'learning rate policy [{config["lr_policy"]}] is not implemented')

    return scheduler

