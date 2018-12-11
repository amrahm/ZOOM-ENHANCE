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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--is_real_time', default=False, type=bool, help='super resolution real time to show')
    parser.add_argument('--delay_time', default=1, type=int, help='super resolution delay time to show')
    parser.add_argument('--model_name', default='epoch_8_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    IS_REAL_TIME = opt.is_real_time
    DELAY_TIME = opt.delay_time
    MODEL_NAME = opt.model_name

    path = 'data/val/SRF_' + str(UPSCALE_FACTOR) + '/data/videos/'
    
    file_names = [join(path, x) for x in listdir(path) if is_image_file(x)]
    file_names.sort()
    print(file_names)
    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    videoWriter = cv2.VideoWriter(out_path + "out.avi", cv2.VideoWriter_fourcc(*'MPEG'), 24, (720, 720))
    for file_name in tqdm(file_names, desc='convert LR videos to HR videos'):
        # videoCapture = cv2.VideoCapture(path + video_name)
        # if not IS_REAL_TIME:
            # fps = videoCapture.get(cv2.CAP_PROP_FPS)
            # size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
            #         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
            # output_name = out_path + video_name.split('.')[0] + '.avi'
            # videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
        videoWriter = cv2.VideoWriter(out_path + "out.avi", cv2.VideoWriter_fourcc(*'MPEG'), 24, (720, 720))
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