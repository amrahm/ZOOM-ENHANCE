import torch
import torch.nn
import torch.nn.functional as F


class AdjacentFrameLoss(torch.nn.Module):
    """
    Frame loss function.
    Computes sum the differences of the difference between the current and next/prev frame of 
    the anchor and the target.
    """

    def __init__(self):
        super(AdjacentFrameLoss, self).__init__()

    def forward(self, a_curr, a_prev, a_next, t_curr, t_prev, t_next):
        a_next_diff = a_curr - a_next
        a_prev_diff = a_curr - a_prev
        t_next_diff = t_curr - t_next
        t_prev_diff = t_curr - t_prev
        loss = torch.nn.L1Loss(a_next_diff, t_next_diff)
        loss = torch.nn.L1Loss(a_prev_diff, t_prev_diff)
        return loss