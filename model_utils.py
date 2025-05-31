

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import torch

def load_model(weight_path, in_channels=6, num_classes=2, device="cpu"):
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    return model
