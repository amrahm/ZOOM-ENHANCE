import argparse
import os
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_video_file
from model import Net

from util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--model', default='epochs/epoch_8_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model

    path = 'data/val/SRF_' + str(UPSCALE_FACTOR) + '/data/videos/'
    
    file_names = [join(path, x) for x in listdir(path) if is_image_file(x)]
    file_names.sort()
    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/' + MODEL_NAME + ".mp4"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    videoWriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'FFV1'), 24, (720, 720))
    for file_name in tqdm(file_names, desc='convert LR videos to HR videos'):
        img = Image.open(file_name).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

        videoWriter.write(out_img)
    videoWriter.release()