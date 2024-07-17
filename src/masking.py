import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from matplotlib import pyplot as plt

class DeepLabV3MobileNetV2(nn.Module):
    def __init__(self):
        super(DeepLabV3MobileNetV2, self).__init__()
        mobilenet_v2_model = mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet_v2_model.features.children()))
        self.classifier = DeepLabHead(1280, 2)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': x}

def mask_fovea(img:torch.Tensor) -> torch.Tensor:
    """
    Fovea masking of a fundus image.

    Args:
        img (torch.Tensor): input RGB image as a torch.Tensor object.

    Returns:
        torch.Tensor: tensor with the fovea masked. 0 values are considered background.    
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 'models/fovea.pth')
    model = torch.load(model_path).to(device)
    
    # Mask fovea
    img = img.unsqueeze(0)
    mask = model(img).get('out')
    mask = torch.argmax(mask, dim=1).squeeze(0)
    
    # Raise error if fovea is not detected
    if mask.sum() == 0:
        raise ValueError('Fovea not detected.')

    return mask

def mask_disc(img:torch.Tensor) -> torch.Tensor:
    """
    Disc masking of a fundus image.

    Args:
        img (torch.Tensor): input RGB image as a torch.Tensor object.

    Returns:
        torch.Tensor: tensor with the disc masked. 0 values are considered background.    
    """
    # Load pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 'models/disc.pth')
    model = torch.load(model_path)
    
    # Mask disc
    img = img.unsqueeze(0)
    mask = model(img)
    mask = torch.argmax(mask, dim=1).squeeze(0)
    
    # Raise error if disc is not detected
    if mask.sum() == 0:
        raise ValueError('Disc not detected.')
    
    return mask