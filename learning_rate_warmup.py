import torch
import bisect

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, lr_decrease_iters=[], last_epoch=-1):

        def lr_lambda(step):
            # warmup
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            else:
                pos = bisect.bisect(lr_decrease_iters, step)
                return 0.1**pos

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)