import os
import gradio as gr
from functools import partial
from typing import List, Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          original_image: Image,
                          xai):
    with globals()[xai](model=HuggingfaceToTensorModelWrapper(model),
                        target_layers=[target_layer],
                        reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image) / 255,
                                              grayscale_cam,
                                              use_rgb=True,
                                              image_weight=0.5)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization, (original_image.size[0], original_image.size[1]))
            results.append(visualization)
        return np.hstack(results)


def clf_run_xai(clf, img, label, xai):
    # Load model
    model, img_tensor, img_pil, original_img = load_model(img, clf)
    target_layer = model.swinv2.layernorm
    targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, label))]

    reshape_transform = partial(swinT_reshape_transform_huggingface,
                                width=img_tensor.shape[2] // 32,
                                height=img_tensor.shape[1] // 32)

    explain_img = Image.fromarray(run_grad_cam_on_image(model=model,
                                                        target_layer=target_layer,
                                                        targets_for_gradcam=targets_for_gradcam,
                                                        reshape_transform=reshape_transform,
                                                        input_tensor=img_tensor,
                                                        input_image=img_pil,
                                                        original_image=original_img,
                                                        xai=xai))

    return explain_img


def clf_pred(clf, img, top_k=3):
    # Load model
    model, img_tensor, img_pil, original_image = load_model(img, clf)
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    # return the label and probability
    predicted_labels_dict = {}
    for i in indices:
        predicted_labels_dict[f"{model.config.id2label[i]}"] = f"{logits.cpu()[0, :].detach().numpy()[i] / 100}"
    return predicted_labels_dict, gr.update(choices=list(predicted_labels_dict.keys()))


def load_model(img, clf):
    # Load model
    model_checkpoint = ""
    if clf == "SwinV2-Tiny":
        model_checkpoint = "microsoft/swinv2-tiny-patch4-window8-256"
    elif clf == "SwinV2-Small":
        model_checkpoint = "microsoft/swinv2-small-patch4-window16-256"
    elif clf == "SwinV2-Base":
        model_checkpoint = "microsoft/swinv2-base-patch4-window8-256"

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_val(image):
        """Apply val_transforms across a batch."""
        img_tensor = val_transforms(image.convert("RGB"))
        img = image.resize((256, 256))
        return img_tensor, img

    img_tensor, img_pil = preprocess_val(img)

    model = AutoModelForImageClassification.from_pretrained(model_checkpoint)

    return model, img_tensor, img_pil, img
