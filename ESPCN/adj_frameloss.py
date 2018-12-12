import torch
import torch.nn


class AdjacentFrameLoss(torch.nn.Module):
    """
    Frame loss function.
    Computes sum the differences of the difference between the current and next/prev frame of 
    the anchor and the target.
    """

    def __init__(self):
        super(AdjacentFrameLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        if torch.cuda.is_available():
            self.l1 = self.l1.cuda()
            self.l2 = self.l2.cuda()

    def forward(self, a_curr, a_prev, a_next, t_curr, t_prev, t_next):
        a_next_diff = a_curr - a_next
        a_prev_diff = a_curr - a_prev
        t_next_diff = t_curr - t_next
        t_prev_diff = t_curr - t_prev
        return self.l1(a_next_diff, t_next_diff) + self.l1(a_prev_diff, t_prev_diff) + self.l2(a_curr, t_curr)