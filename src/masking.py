import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from loguru import logger
from definitions import MODELS_DIR
from torchvision.models import mobilenet_v2
from skimage.measure import label, regionprops_table
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

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

def mask_part(img: torch.Tensor, model_path: str, expected_size: tuple = (512, 512)) -> tuple:
    """
    Generic masking function for a part of a fundus image.

    Args:
        img (torch.Tensor): input RGB image as a torch.Tensor object.
        model_path (str): path to the pre-trained model.
        expected_size (tuple): expected size of the output mask.

    Returns:
        bool: True if the part is detected, False otherwise.
        torch.Tensor: tensor with the part masked. 0 values are considered background.
    """
    # Init
    model_part = model_path.split("/")[-1].split(".")[0].capitalize()
    
    # Load pre-trained model
    model = torch.load(model_path)
    
    # Mask part
    img = img.unsqueeze(0)
    mask = model(img)
    mask = mask.get('out') if isinstance(mask, dict) else mask
    mask = torch.argmax(mask, dim=1).squeeze(0)
    
    # Log error if part is not detected
    flag = mask.any()
    log_level = logger.trace if flag else logger.error
    log_message = f'{model_part} segmentation..........<green>SUCCESS</green>' if flag else f'{model_part} segmentation..........<red>FAILED</red>'
    log_level(log_message)
    
    # Resize mask to expected size if needed
    if mask.shape != torch.Size(expected_size):
        mask = nn.functional.interpolate(mask.to(torch.uint8).unsqueeze(0).unsqueeze(0), 
                                         size=expected_size, mode='nearest').squeeze(0).squeeze(0)

    return flag, mask

def generate_masks(img: Image) -> dict:
    """Generate masks for the disc, cup and fovea.

    Args:
        img (Image): input image as a PIL Image object.

    Returns:
        dict: dictionary containing the masks for the disc, cup and fovea.
    """
    # Convert image to tensor for processing
    transform_disc = Compose([Resize((512, 512)), ToTensor(), Normalize((0.5,), (0.5,))])
    transform_fovea = Compose([Resize((224, 224)), ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img4fovea= transform_fovea(img).to(device)
    img4disc_cup = transform_disc(img).to(device)
    
    # Process image
    _, mask_d = mask_part(img4disc_cup, os.path.join(MODELS_DIR, 'disc.pth'))
    _, mask_c = mask_part(img4disc_cup, os.path.join(MODELS_DIR, 'cup.pth'))
    _, mask_f = mask_part(img4fovea, os.path.join(MODELS_DIR, 'fovea.pth'), expected_size=(224, 224))
    
    # Prepare output
    masks = {
        'disc': mask_d,
        'cup': mask_c,
        'fovea': mask_f
        }

    output = {key: mask.to(torch.uint8).cpu().numpy() if torch.any(mask) else None for key, mask in masks.items()}

    return output

def clean_segmentations(masks: dict) -> dict:
    """Clean segmentations by removing small areas and keeping the roundest one.

    Args:
        masks (dict): dictionary containing the masks for the disc, cup and fovea.

    Returns:
        dict: dictionary containing the cleaned masks for the disc, cup and fovea.
    """
    # Initialize dictionary to store cleaned masks
    cleaned_masks = {}
    
    for key, mask in masks.items():
        
        # Open and close operations to remove small areas
        cleaned_mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
        cleaned_mask = cv2.morphologyEx(cleaned_mask1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Label and compute roundness
        cleaned_mask_labelled = label(cleaned_mask)
        props = regionprops_table(cleaned_mask_labelled, properties=('area', 'perimeter'))
        radii = props.get('perimeter') / (2 * np.pi) + 0.5
        roundness = [(4 * np.pi * props.get('area')[idx] / (props.get('perimeter')[idx] ** 2)) * (1 - 0.5 * r)**2 
                     if props.get('perimeter')[idx] != 0 else 0.0 for idx, r in enumerate(radii)]
        
        if len(roundness) != 0:
            # Remove areas smaller than 50 px
            roundness = np.array(roundness)
            roundness[props.get('area') < 50] = 0
            # Keep the roundest area
            idx = np.argmax(roundness)
            cleaned_mask = (cleaned_mask_labelled == idx + 1)
            
        # Store cleaned mask in dictionary
        cleaned_masks[key] = cleaned_mask.astype(np.uint8)
        
    return cleaned_masks

def merge_masks(masks: dict) -> np.ndarray:
    """Merge binary masks into a single mask.

    Args:
        masks (dict): dictionary containing the masks to be merged. All masks must have the same shape. 0 values are considered background.

    Returns:
        np.ndarray: unified mask.
    """
    unified_mask = np.zeros_like(next(iter(masks.values()))).astype(float)
    
    for idx, mask in enumerate(masks.values()):
        
        unified_mask[mask.astype(bool)] = idx + 1
        
    return unified_mask
