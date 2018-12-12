import argparse
import os
from os import listdir
from os.path import join

import cv2
import skvideo.io
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange

from data_utils import is_video_file
from model import Net
from model2 import TwoNet

from util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--model', default='epochs/epoch_8_100.pt', type=str, help='super resolution model name')
    parser.add_argument('--stacked', default=False, type=bool, help='super resolution model name')
    parser.add_argument('--get_full', default=False, type=bool, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model
    STACKED = opt.stacked
    GET_FULL = opt.get_full

    if GET_FULL:
        path = 'data/val/SRF_' + str(UPSCALE_FACTOR) + '/target/videos/'
    else:
        path = 'data/val/SRF_' + str(UPSCALE_FACTOR) + '/data/videos/'

    file_names = [join(path, x) for x in listdir(path) if is_image_file(x)]
    file_names.sort()

    if not GET_FULL:
        if STACKED:
            model = TwoNet(upscale_factor=UPSCALE_FACTOR)
        else:
            model = Net(upscale_factor=UPSCALE_FACTOR)

        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(MODEL_NAME))



    if GET_FULL:
        out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/' + "FULLRES" + ".avi"
    else:
        out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/' + MODEL_NAME + ".avi"
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    # videoWriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'LAGS'), 24, (720, 720))
    videoWriter = skvideo.io.FFmpegWriter(out_path,
        outputdict={
            '-vcodec': 'libx264',  #use the h.264 codec
            '-crf': '0',  #set the constant rate factor to 0, which is lossless
            '-preset':
                'veryslow'  #the slower the better compression, in princple, try 
            #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        })
    for i in tqdm(trange(len(file_names) - 2), desc='convert LR videos to HR videos'):
        file_name = file_names[i]
        img = Image.open(file_name).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if GET_FULL:
            out = image
        else:
            if torch.cuda.is_available():
                image = image.cuda()

            if STACKED:
                file_name2 = file_names[i + 1]
                img2 = Image.open(file_name2).convert('YCbCr')
                y2, cb2, cr2 = img2.split()
                image2 = Variable(ToTensor()(y2)).view(1, -1, y2.size[1], y2.size[0])
                if torch.cuda.is_available():
                    image2 = image2.cuda()
                stacked = torch.cat((image, image2), 1)
                out = model(stacked)
            else:
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

        # videoWriter.write(out_img)
        videoWriter.writeFrame(out_img[:,:,::-1])  #write the frame as RGB not BGR
    # videoWriter.release()
    videoWriter.close()