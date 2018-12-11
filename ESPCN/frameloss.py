import torch
import torch.nn
import torch.nn.functional as F


class FrameLoss(torch.nn.Module):
    """
    Frame loss function.
    """

    def __init__(self):
        super(FrameLoss, self).__init__()

    def forward(self, a_curr, a_next, t_curr, t_next):
        c_diff = a_curr - a_next
        t_diff = t_curr - t_next
        loss = torch.nn.MSELoss(c_diff, t_diff)
        return loss