colors = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
    (255, 0, 0),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

torch_device = 'cpu'  # Change to 'cpu' or 'cuda:0' to suit your system.

# Load the label of COCO dataset
coco = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant',
        'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe',
        'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'N/A', 'dining table',
        'N/A', 'N/A', 'toilet',
        'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator',
        'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

CLASSIFICATION_XAI = ["GradCAM", "GradCAMPlusPlus", "EigenCAM", "EigenGradCAM", "HiResCAM"]
SEGMENTATION_XAI = ["GradCAM", "GradCAMPlusPlus", "EigenCAM", "EigenGradCAM", "ScoreCAM", "HiResCAM", "AblationCAM",
                    "XGradCAM"]
DETECTION_XAI = ["D-RISE", "D-CLOSE", "GCAME"]

MODEL_NAME = "gpt-4"
OPENAI_API_KEY = "REPLACE_YOUR_API"

IMAGENETV2_TEST_PATH = "images/imagenet/imagenetv2-matched-frequency-format-val"
SERVER_PORT = 7860
