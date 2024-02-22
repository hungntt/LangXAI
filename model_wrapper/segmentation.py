import json

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from config import TTPLA_LABEL_PATH
from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper
from model.segmentation.semantic_segmentation_target import SemanticSegmentationTarget


def seg_run_pred(seg_model, category, image):
    segmentation_image = seg_model.predict(category=category)
    return segmentation_image


def seg_run_xai(seg_model, xai):
    explanation = seg_model.explain(xai)
    return explanation


def seg_get_label(model_name, image, image_id, cat):
    seg_model = SegmentationModule(model=model_name, image=image)
    label = seg_model.label(image_id, cat)
    return label, seg_model


class SegmentationModule:
    def __init__(self, model='ResNet101', image=None):
        self.image = None
        self.mask_float = None
        self.input_tensor = None
        self.rgb_img = None
        self.category_idx = None
        self.sem_classes = [
            '__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'
        ]

        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.sem_classes)}

        if model == 'ResNet101':
            self.model = deeplabv3_resnet101(pretrained=False, num_classes=len(self.sem_classes))
        elif model == 'ResNet50':
            self.model = torch.hub.load('pytorch/vision:v0.11.0', 'deeplabv3_resnet50', pretrained=False,
                                        num_classes=len(self.sem_classes))

        PATH = f'model/segmentation/model_{model}.pth'
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(PATH))
        else:
            self.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        self.model = self.model.eval()

        scripted_module = torch.jit.script(self.model)
        self.model = optimize_for_mobile(scripted_module)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)
        self.image = image

    def predict(self, category):
        img_arr = np.array(self.image)
        self.rgb_img = np.float32(img_arr) / 255
        self.input_tensor = preprocess_image(self.rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if torch.cuda.is_available():
            self.input_tensor = self.input_tensor.cuda()

        output = self.model(self.input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
        self.category_idx = self.sem_class_to_idx[category]
        mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        mask_uint8 = 255 * np.uint8(mask == self.category_idx)
        self.mask_float = np.float32(mask == self.category_idx)

        # Resize mask to the same size as input image
        mask_uint8 = cv2.resize(mask_uint8, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert mask to color image
        mask_color = np.zeros_like(img_arr)
        mask_color[mask_uint8 > 0] = (0, 0, 255)  # set the color of the segmentation mask to red (BGR format)
        # Overlay segmentation mask on input image
        bgr_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.addWeighted(bgr_image, 0.5, mask_color, 0.5, 0)

        # Convert the overlay image back to RGB for PIL
        rgb_overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

        # Convert the overlay image to a PIL image object
        segmentation = Image.fromarray(rgb_overlayed_image)
        return segmentation

    def explain(self, xai):
        target_layers = [self.model.model.backbone.layer4]
        targets = [SemanticSegmentationTarget(self.category_idx, self.mask_float)]

        with globals()[xai](model=self.model, target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=self.input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(self.rgb_img, grayscale_cam, use_rgb=True)
        explanation = Image.fromarray(cam_image)

        return explanation

    def label(self, image_id, cat):
        self.label_path = TTPLA_LABEL_PATH + f'/{image_id}.json'
        with open(self.label_path, 'r') as f:
            data = json.load(f)
        mask = Image.new('RGBA', self.image.size, (0, 0, 0, 0))

        draw = ImageDraw.Draw(mask)
        for shape in data['shapes']:
            if shape['label'] == cat:
                points = [(p[0], p[1]) for p in shape['points']]
                try:
                    draw.polygon(points, fill=tuple(shape['fill_color']), outline=tuple(shape['line_color']))
                except TypeError:
                    # fill red color if no fill color is specified
                    draw.polygon(points, fill=(255, 0, 0, 255), outline=(0, 255, 0, 255))
                # Add the label name to the annotation
                try:
                    draw.text(points[0], shape['label'], fill=tuple(shape['fill_color']), align='center')
                except TypeError:
                    draw.text(points[0], shape['label'], fill=(255, 0, 0, 255), align='center')

        # Combine the image and the segmentation mask
        result = Image.alpha_composite(self.image.convert('RGBA'), mask)

        # Convert the result to a PIL image object
        coco_image = result.convert('RGB')

        return coco_image
