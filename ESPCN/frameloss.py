import torch
import torch.nn
import torch.nn.functional as F


class FrameLoss(torch.nn.Module):
    """
    Frame loss function.
    """

    def __init__(self):
        super(FrameLoss, self).__init__()

    def forward(self, a_curr, a_prev, t_curr, t_prev):
        c_diff = a_curr - a_prev
        t_diff = t_curr - t_prev
        loss = torch.nn.L1Loss(c_diff, t_diff)
        return loss