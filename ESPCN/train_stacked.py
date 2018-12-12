import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import DatasetFromFolderVideos
from model2 import TwoNet
from psnrmeter import PSNRMeter
from frameloss import FrameLoss
from util import *


def processor(sample):
    def tensorFromSample(key):
        tensor = Variable(sample[0][key])
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    image = tensorFromSample("image")
    next_image = tensorFromSample("next_image")
    target = tensorFromSample("target")
    # next_target = tensorFromSample("next_target")

    stacked = torch.cat((image, next_image), 1)
    a_curr = model(stacked)
    # loss = criterion(a_curr, a_next, target, next_target)
    loss = criterion(a_curr, target)

    return loss, a_curr


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][0]['target'])
    meter_loss.add(state['loss'].data.item())


def on_start_epoch(state):
    reset_meters()
    scheduler.step()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())

    reset_meters()

    engine.test(processor, val_loader)
    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value())

    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), EPOCH_DIR + 'epoch_%d_%d.pt' % (UPSCALE_FACTOR, state['epoch']))


if __name__ == "__main__":
    EPOCH_DIR = "epochs_stacked/"
    makePathIfNotExists(EPOCH_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = DatasetFromFolderVideos('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolderVideos('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=12, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=12, batch_size=64, shuffle=False)

    model = TwoNet(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
