import io
import timeit

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms as T

from config import coco, torch_device
from xai.gcame import GCAME

overlap = {name for name in mcolors.CSS4_COLORS if f'xkcd:{name}' in mcolors.XKCD_COLORS}
all_colors = []

# Load pretrained Faster-RCNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

for color_name in overlap:
    css4 = mcolors.CSS4_COLORS[color_name]
    all_colors.append(css4)
    xkcd = mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper()
    all_colors.append(xkcd)


def get_prediction(pred, threshold):
    """
    get_prediction
      Parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      Method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    pred_class = pred[0]['labels'].cpu().numpy()
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t) == 0:
        flag = 0.
        return flag
    else:
        pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    scores = pred_score[:pred_t + 1]
    return pred_boxes, pred_class, scores


def det_run_xai(det_model, det_xai, img):
    """
    det_run_xai: Run XAI for detection task
    :param det_model: Detection model
    :param det_xai: XAI method for detection task
    :param img: input image
    :return: explanation map
    """
    img_arr = np.array(img)

    rgb_img = np.float32(img_arr) / 255
    if det_model == 'FasterRCNN':
        # Load image
        transform = T.Compose([T.ToTensor()])
        org_h, org_w, _ = rgb_img.shape

        # Get prediction
        model.to(torch_device)
        inp = transform(rgb_img)
        prediction = model([inp.to(torch_device)])
        rs = get_prediction(prediction, 0.5)

        # Show prediction
        boxes, pred_cls, pred = rs
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)
        for i in range(len(boxes)):
            x_min, y_min = boxes[i][0]
            x_max, y_max = boxes[i][1]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1,
                                     edgecolor=all_colors[pred_cls[i]], facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, coco[pred_cls[i]], style='italic',
                    bbox={'facecolor': all_colors[pred_cls[i]],
                          'alpha': 0.5, })

        for i in range(len(boxes)):
            boxes[i].append(pred_cls[i])
            boxes[i].append(pred[i])

        # Return the prediction image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        prediction = Image.open(img_buf)

        target_layer = [
            # 'backbone.fpn.inner_blocks.0.0',
            'backbone.fpn.layer_blocks.0.0',

            # 'backbone.fpn.inner_blocks.1.0',
            'backbone.fpn.layer_blocks.1.0',

            'backbone.fpn.inner_blocks.2.0',
            'backbone.fpn.layer_blocks.2.0',

            # 'backbone.fpn.inner_blocks.3.0',
            'backbone.fpn.layer_blocks.3.0',
        ]

        idx = 0
        if det_xai == 'GCAME':
            cam = GCAME(model, target_layer)
            inp = inp.to('cpu')
            start = timeit.default_timer()
            out = cam(inp, [boxes[idx]], index=idx)
            stop = timeit.default_timer()

            print("Time: {}s".format(stop - start))
            box = boxes[idx]
            (x_min, y_min), (x_max, y_max), cls, score = box
            fig, ax = plt.subplots()
            ax.imshow(rgb_img)
            rect = patches.Rectangle((x_min, y_min),
                                     x_max - x_min,
                                     y_max - y_min,
                                     linewidth=1,
                                     edgecolor=all_colors[cls],
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, coco[cls], style='italic', bbox={'facecolor': all_colors[cls], 'alpha': 0.5})
            ax.imshow(out, cmap='jet', alpha=0.5)
            # No white space and remove axis
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.axis('off')
            # Return the fig as a PIL Image
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            explanation = Image.open(img_buf)
        return prediction, explanation
