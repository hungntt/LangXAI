import os

from config import TTPLA_IMAGE_PATH
from utils import load_all_images_in_a_folder


def ttpla_images_loader(path=TTPLA_IMAGE_PATH):
    images_and_labels = []
    for sub_folder_name in os.listdir(path):
        # Get the label of the image based on filename
        try:
            label = sub_folder_name
            if os.path.isdir(os.path.join(path, sub_folder_name)):
                images_path = load_all_images_in_a_folder(os.path.join(path, sub_folder_name))
                for image_path in images_path:
                    image_id = image_path.split('/')[-1].split('.')[0]
                    images_and_labels.append([image_path, label, image_id])
        except Exception as e:
            print(sub_folder_name, e)
    return images_and_labels
