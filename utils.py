import base64
import os
from io import BytesIO

import torch


def load_all_images_in_a_folder(path):
    images = []
    for filename in os.listdir(path):
        # If filename end with .jpg or .png
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            images.append(os.path.join(path, filename))
    return images


def convert_pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def check_devices_torch():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def extract_index_from_object_detection_output(output: gr.SelectData):
    """
    Extract number 0 from "person_0" for example
    """
    return output.index
