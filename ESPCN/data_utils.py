import argparse
import os
from os import listdir
from os.path import join

from PIL import Image
import pims
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import tqdm
import pymp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, videos=True, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        vids = "/videos" if videos else "/images"
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data' + vids 
        self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/target'+ vids 
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
        target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_filenames)


def generate_dataset(data_type, upscale_factor):
    makePathIfNotExists('data/dataset/videos/' + data_type)
    videos_name = [x for x in listdir('data/dataset/videos/' + data_type) if is_video_file(x)]

    root = 'data/' + data_type
    makePathIfNotExists(root)
    path = root + '/SRF_' + str(upscale_factor)
    makePathIfNotExists(path)
    image_path = path + '/data'
    makePathIfNotExists(image_path + '/videos/')
    target_path = path + '/target'
    makePathIfNotExists(target_path + '/videos/')

    # with pymp.Parallel(24) as p:
        # for i in tqdm(p.range(len(videos_name)), desc='generate ' + data_type + ' video dataset with upscale factor = '
        #         + str(upscale_factor) + ' from dataset'):
    for i in tqdm(range(len(videos_name)), desc='generate ' + data_type + ' video dataset with upscale factor = '
            + str(upscale_factor) + ' from dataset'):
        # import pdb; pdb.set_trace()
        video_name = videos_name[i]
        video_rel_path = 'data/dataset/videos/' + data_type + '/' + video_name
        target_rel_path = target_path + '/videos/' + video_name
        ffmpeg_extract_subclip(video_rel_path, 0, 5, targetname=target_rel_path)
        video = pims.open(os.path.abspath(target_rel_path))
        #image_path
        ffmpeg_resize(video, image_path + '/videos/' + video_name,str(video[0].shape))

def makePathIfNotExists(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--set', default='train', type=str, help='\'train\' or \'val\' specifies generating for that set')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    SET = opt.set

    generate_dataset(data_type=SET, upscale_factor=UPSCALE_FACTOR)
