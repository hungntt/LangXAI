import os

from images.imagenet.imagenet_label import imagenet_label
from utils import load_all_images_in_a_folder

from config import IMAGENETV2_TEST_PATH


def imagenetv2_images_loader(path=IMAGENETV2_TEST_PATH):
    images_and_labels = []
    for filename in os.listdir(path):
        # Get the label of the image based on filename
        try:
            label = imagenet_label[int(filename)]
            if os.path.isdir(os.path.join(path, filename)):
                images_path = load_all_images_in_a_folder(os.path.join(path, filename))
                for image_path in images_path:
                    images_and_labels.append([image_path, label])
        except Exception as e:
            print(filename, e)
    return images_and_labels
