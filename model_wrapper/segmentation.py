import json

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper
from model.segmentation.semantic_segmentation_target import SemanticSegmentationTarget


def seg_run_xai(model_name, xai, category, image):
    # SegmentationExplainer code here
    explainer = SegmentationExplainer(model=model_name)
    segmentation_image, cam_image = explainer.explain(image=image,
                                                      category=category,
                                                      xai=xai)
    return segmentation_image, cam_image


class SegmentationExplainer:
    def __init__(self, model='ResNet101'):
        self.sem_classes = [
            '__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'
        ]

        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.sem_classes)}

        if model == 'ResNet101':
            self.model = deeplabv3_resnet101(pretrained=False, num_classes=len(self.sem_classes))
        elif model == 'ResNet50':
            self.model = deeplabv3_resnet50(pretrained=False, num_classes=len(self.sem_classes))

        PATH = f'model/segmentation/model_{model}.pth'
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(PATH))
        else:
            self.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        self.model = self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)

    def explain(self, image, category, xai):
        img_arr = np.array(image)

        rgb_img = np.float32(img_arr) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # # ======== Label ========
        # if label_path is not None:
        #     with open(label_path, 'r') as f:
        #         data = json.load(f)
        #     mask = Image.new('RGBA', orig_image.size, (0, 0, 0, 0))
        #
        #     draw = ImageDraw.Draw(mask)
        #     for shape in data['shapes']:
        #         if shape['label'] == category:
        #             points = [(p[0], p[1]) for p in shape['points']]
        #             try:
        #                 draw.polygon(points, fill=tuple(shape['fill_color']), outline=tuple(shape['line_color']))
        #             except TypeError:
        #                 # fill red color if no fill color is specified
        #                 draw.polygon(points, fill=(255, 0, 0, 255), outline=(0, 255, 0, 255))
        #             # Add the label name to the annotation
        #             try:
        #                 draw.text(points[0], shape['label'], fill=tuple(shape['fill_color']), align='center')
        #             except TypeError:
        #                 draw.text(points[0], shape['label'], fill=(255, 0, 0, 255), align='center')
        #
        #     # Combine the image and the segmentation mask
        #     result = Image.alpha_composite(orig_image.convert('RGBA'), mask)
        #
        #     # Convert the result to a PIL image object
        #     coco_image = result.convert('RGB')
        # else:
        #     coco_image = None

        # ======= Model output =======

        output = self.model(input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
        category_idx = self.sem_class_to_idx[category]
        mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        mask_uint8 = 255 * np.uint8(mask == category_idx)
        mask_float = np.float32(mask == category_idx)

        # Resize mask to the same size as input image
        mask_uint8 = cv2.resize(mask_uint8, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert mask to color image
        mask_color = np.zeros_like(img_arr)
        mask_color[mask_uint8 > 0] = (0, 0, 255)  # set the color of the segmentation mask to red (BGR format)
        # Overlay segmentation mask on input image
        bgr_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.addWeighted(bgr_image, 0.5, mask_color, 0.5, 0)

        # Convert the overlayed image back to RGB for PIL
        rgb_overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

        # Convert the overlayed image to a PIL image object
        segmentation = Image.fromarray(rgb_overlayed_image)

        target_layers = [self.model.model.backbone.layer4]
        targets = [SemanticSegmentationTarget(category_idx, mask_float)]

        with globals()[xai](model=self.model, target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        explanation = Image.fromarray(cam_image)

        return segmentation, explanation
