import torch.nn as nn
import torch


class TwoNet(nn.Module):
    def __init__(self, upscale_factor):
        super(TwoNet, self).__init__()

        self.conv1 = nn.Conv2d(2, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        return x


if __name__ == "__main__":
    model = TwoNet(upscale_factor=8)
    print(model)