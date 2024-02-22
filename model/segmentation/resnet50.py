from torchvision.models.segmentation import deeplabv3_resnet50


class ResNet50:
    def __init__(self):
        self.model = deeplabv3_resnet50(pretrained=True, progress=False)

