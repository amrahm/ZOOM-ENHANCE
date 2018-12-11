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



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

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
    def __init__(self, dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/videos' 
        self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/videos' 
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

class DatasetFromFolderVideos(Dataset):
    def __init__(self, dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolderVideos, self).__init__()
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/videos'
        self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/videos'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.image_filenames.sort()
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.target_filenames.sort()
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if index >= len(self.image_filenames) - 2:
            index = 0

        
        frame_no = self.image_filenames[index].split(".")[-2]
        next_frame_no = self.image_filenames[index + 1].split(".")[-2]
        if int(next_frame_no) != int(frame_no) + 1:
            index = index + 1

        image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
        next_image, _, _ = Image.open(self.image_filenames[index + 1]).convert('YCbCr').split()
        target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        next_target, _, _ = Image.open(self.target_filenames[index+1]).convert('YCbCr').split()
        if self.input_transform is not None:
            image = self.input_transform(image)
            next_image = self.input_transform(next_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            next_target = self.target_transform(next_target)

        return {
            'image': image,
            'next_image': next_image,
            'target': target,
            'next_target': next_target
        }

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

    with pymp.Parallel(24) as p:
        for i in tqdm(p.range(len(videos_name)), desc='generate ' + data_type + ' video dataset with upscale factor = '
                + str(upscale_factor) + ' from dataset'):
            video_name = videos_name[i]
            video = pims.open('data/dataset/videos/' + data_type + '/' + video_name)
            try:
                frame_no = 1
                for image in video[60:90]: #Save frames 60 to 90 only
                    image = Image.fromarray(image) #convert pims frame to PIL image
                    target = image.copy()
                    
                    crop_size = calculate_valid_crop_size(720, upscale_factor)
                    lr_transform = input_transform(crop_size, upscale_factor)
                    hr_transform = target_transform(crop_size)

                    image = lr_transform(image)
                    target = hr_transform(target)

                    image_name = video_name.replace(video_name.split(".")[-1], "") + str(frame_no).zfill(2) + ".png"
                    image.save(image_path + '/videos/' + image_name)
                    target.save(target_path + '/videos/' + image_name)
                    frame_no += 1
            except (pims.api.UnknownFormatError, IndexError) as e:
                print(e)

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





# a = ["W.01.png", "W.02.png", "W.20.png", "W.10.png", "W.11.png"]
# a.sort()
# print(a)