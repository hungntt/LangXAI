import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from torch.nn import functional as F
import torch.nn as nn

from math import floor
import math
import matplotlib.colors as mcolors
from tqdm import tqdm
import copy
import timeit

from config import coco


class GCAME(object):
    def __init__(self, model, target_layer, img_size=(640, 640), **kwargs):
        """
        Parameters:
          - model: The model in nn.Module() to analyze
          - target_layers: List of names of the target layers in model.named_modules()
          - img_size: The size of image in tuple
        Variables:
          - self.gradients, self.activations: Dictionary to save the value when
            do forward/backward in format {'name_layer': activation_map/gradient}
          - self.handlers: List of hook functions
        """
        self.model = model.eval()
        self.img_size = img_size
        self.gradients = dict()
        self.activations = dict()
        self.target_layer = target_layer
        self.handlers = []

        def save_grads(key):
            def backward_hook(module, grad_inp, grad_out):
                self.gradients[key] = grad_out[0].detach()

            return backward_hook

        def save_fmaps(key):
            def forward_hook(module, inp, output):
                self.activations[key] = output

            return forward_hook

        for name, module in list(self.model.named_modules())[1:]:

            if name in self.target_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def __call__(self, img, box, index=None):
        return self.forward(img, box, index)

    # This function to create Gaussian mask used in algorithm
    @staticmethod
    def create_heatmap(output_width, output_height, p_x, p_y, sigma):
        """
        Parameters:
          - output_width, output_height: The kernel size of Gaussian mask
          - p_x, p_y: The center of Gaussian mask
          - sigma: The standard deviation of Gaussian mask
        Returns:
          - mask: The 2D-array Gaussian mask in range [0, 1]
        """
        X1 = np.linspace(1, output_width, output_width)
        Y1 = np.linspace(1, output_height, output_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - floor(p_x)
        Y = Y - floor(p_y)
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma ** 2
        Exponent = D2 / E2
        mask = np.exp(-Exponent)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask

    def forward(self, img, box, index=None):
        """
        Parameters:
          - img: input image in Tensor[3, H, W]
          - box: The bounding box to analyze List[(xmin, ymin), (xmax, ymax), cls, score]
          - index: The index of target bounding box in int
        Returns:
          - score_saliency_map: The saliency map of target object
        """

        c, h, w = img.size()
        org_size = (h, w)

        # Get input image size
        transform_img = self.model.transform([img])[0]
        self.img_size = transform_img.image_sizes[0]
        self.model.zero_grad()

        # Get prediction
        output = self.model([img])
        output[0]['scores'][index].backward(retain_graph=True)

        # Create saliency map
        score_saliency_map = np.zeros((org_size))

        for target_layer in self.target_layer:
            map = self.activations[target_layer]
            grad = self.gradients[target_layer]

            # Select the branch that the target comes out
            if grad.max() == 0 and grad.min() == 0:
                continue

            map = map.squeeze().detach().cpu().numpy()
            grad = grad.squeeze().detach().cpu().numpy()

            # Calculate the proportion between the input image and the gradient map
            stride = math.sqrt((self.img_size[0] * self.img_size[1]) / (grad.shape[1] * grad.shape[2]))

            for j in tqdm(range(map.shape[0])):
                new_map = copy.deepcopy(map[j])
                pos_grad = copy.deepcopy(grad[j])
                neg_grad = copy.deepcopy(grad[j])

                # Get the positive part of gradient map
                pos_grad[pos_grad < 0] = 0
                mean_pos_grad = np.mean(pos_grad)
                max_grad = pos_grad.max()
                idx, idy = (pos_grad == max_grad).nonzero()
                if len(idx) == 0 or len(idy) == 0 or mean_pos_grad == 0:
                    continue

                idx = idx[0]
                idy = idy[0]
                kn_size = math.floor((math.sqrt(grad.shape[1] * grad.shape[2]) - 1) / 2) / 3
                pos_sigma = (np.log(abs(mean_pos_grad)) / kn_size) * np.log(stride)

                pos_sigma = max(abs(pos_sigma), 1.)
                pos_mask = self.create_heatmap(grad[j].shape[1], grad[j].shape[0], idy, idx, pos_sigma)
                pos_weighted_map = (new_map * mean_pos_grad) * pos_mask

                # Get the negative part of gradient map
                neg_grad[neg_grad > 0] = 0
                mean_neg_grad = np.mean(neg_grad)
                if mean_neg_grad == 0:
                    continue

                min_grad = np.unique(neg_grad[neg_grad != 0])[-1]
                idx_, idy_ = (neg_grad == min_grad).nonzero()
                if len(idx_) == 0 or len(idy_) == 0:
                    continue

                idx_ = idx_[0]
                idy_ = idy_[0]
                neg_sigma = (np.log(abs(mean_neg_grad)) / kn_size) * np.log(stride)
                neg_sigma = max(abs(neg_sigma), 1.)
                neg_mask = self.create_heatmap(grad[j].shape[1], grad[j].shape[0], idy_, idx_, neg_sigma)
                neg_weighted_map = (new_map * mean_neg_grad) * neg_mask

                # Sum up the weighted feature map
                weighted_map = pos_weighted_map - neg_weighted_map
                weighted_map = cv2.resize(weighted_map, (org_size[1], org_size[0]))
                weighted_map[weighted_map < 0.] = 0.
                score_saliency_map += weighted_map

        score_saliency_map = (score_saliency_map - score_saliency_map.min()) / (
                score_saliency_map.max() - score_saliency_map.min())
        return score_saliency_map
