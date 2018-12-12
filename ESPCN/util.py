import os

def makePathIfNotExists(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])

