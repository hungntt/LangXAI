from PIL import Image

import skimage.io as skio
import io
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import os

dataDir = 'images'
dataType = 'val2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
coco = COCO(annFile)


def load_coco_images():
    images_and_id = []
    # Get 10 images from the dataset
    imgIds = coco.getImgIds()
    imgIds = imgIds[:10]
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        images_and_id.append([img['coco_url'], imgId])
    return images_and_id


def load_coco_annotations(img_id):
    # get all images containing given categories, select one at random
    img_id = int(img_id)
    img = coco.loadImgs(img_id)[0]
    I = skio.imread(img['coco_url'])
    fig, ax = plt.subplots(1, figsize=(12, 12))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=True)
    for i, ann in enumerate(anns):
        ax.text(ann['bbox'][0], ann['bbox'][1] - 2, '%s' % (coco.loadCats(ann['category_id'])[0]['name']),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    # Remove axis and padding and white space
    ax.axis('off')
    ax.set_ylim(I.shape[0], 0)
    ax.set_xlim(0, I.shape[1])
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    ann = Image.open(img_buf)
    return ann
