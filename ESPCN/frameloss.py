import torch
import torch.nn
import torch.nn.functional as F


class FrameLoss(torch.nn.Module):
    """
    Frame loss function.
    """

    def __init__(self):
        super(FrameLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        if torch.cuda.is_available():
            self.l1 = self.l1.cuda()
            self.l2 = self.l2.cuda()

    def forward(self, a_curr, a_next, t_curr, t_next):
        a_next_diff = a_curr - a_next
        t_next_diff = t_curr - t_next
        return self.l1(a_next_diff, t_next_diff) + self.l2(a_curr, t_curr)