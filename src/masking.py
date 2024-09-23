import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from loguru import logger
from definitions import MODELS_DIR
import scipy.spatial.distance as distance
from torchvision.models import mobilenet_v2
from skimage.measure import label, regionprops_table
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from src.geom import is_ellipse_contained, intersection_line_ellipse
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

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
    log_level = logger.debug if flag else logger.error
    log_message = f'{model_part} segmentation... SUCCESS' if flag else f'{model_part} segmentation... FAILED'
    log_level(log_message)
    
    # Raise error if part is not detected
    if not flag:
        raise ValueError(f'{model_part} segmentation failed.')
    
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
    _, mask_f = mask_part(img4fovea, os.path.join(MODELS_DIR, 'fovea.pth'))
    
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
    # Create multi-dimensional unified mask by sticking masks together along the third axis
    # Number of masks
    n_masks = len(masks)
    # Masks shape
    mask_shape = np.unique([mask.shape for mask in masks.values()], axis=0)
    if mask_shape.shape[0] != 1:
        logger.error('All masks must have the same shape.')
        raise ValueError('All masks must have the same shape.')
    # Initialize unified mask
    unified_mask = np.zeros((mask_shape[0][0], mask_shape[0][1], n_masks), dtype=np.uint8)
    
    # Fill unified mask
    for idx, mask in enumerate(masks.values()):
        unified_mask[..., idx] = mask * (idx + 1)
    
    return unified_mask

def cdr_profile(mask:np.ndarray, ang_step:int=15) -> list:
    """
    Compute the cup-to-disc ratio profile of a fundus image.
    
    Parameters
    ----------
    mask : np.ndarray
        Greyscale mask of the fundus image. We assume pixels with value 0 are background.
        
    ang_step : int
        Angular step size in degrees. Values must be in the range [1, 180].
    
    Returns
    -------
    list
        List containing the intersection points of the line passing through the centre of the cup with the ellipses, and the cup-to-disc ratio profile.
    """
    # Check mask
    assert mask.ndim == 2, "Mask must be a 2D array."
    assert np.unique(mask).size >= 3, "Mask must contain at least 3 unique pixel values, 0 being 'Background', 1 'Disc', 2 'Cup'."
    
    # Check angular step
    assert ang_step >= 1, "Angular step must be greater than or equal to 1."
    assert ang_step <= 180, "Angular step must be less than or equal to 180."
    assert np.mod(180, ang_step) == 0, "180 must be divisible by the angular step."
    
    # Compute ellipses for pixel values 1 and 2 (disc and cup)
    ellipses = []
    for i in [(1, 2), (2, 2)]:
        # Make copy of mask
        tmp_mask = np.zeros_like(mask, dtype=np.uint8)
        tmp_mask[(mask == i[0]) | (mask == i[1])] = 255
        cnt, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse = cv2.fitEllipse(cnt[0])
        ellipses.append(ellipse)
        
    # Sort ellipses by area, smallest (cup) to largest (disc)
    ellipses = sorted(ellipses, key=lambda x: x[1][0]*x[1][1])
    
    # Check if ellipses are contained within each other
    if not is_ellipse_contained(ellipses[0], ellipses[1]):
        logger.error('Cup ellipse is not contained within disc ellipse.')
        raise ValueError('Cup ellipse is not contained within disc ellipse.')
    else:
        logger.debug('Cup ellipse is contained within disc ellipse.')
    
    # Calculate centre as the midpoint between cup and disc ellipses centres
    centre = ((np.array(ellipses[0][0]) + np.array(ellipses[1][0])) / 2).reshape(-1,1)
    x0, y0 = centre[:,0]
    
    # Intersection line with ellipses
    m = np.array([np.tan(np.deg2rad(ang)) for ang in np.arange(0, 180, ang_step)])
    n = np.array([y0 - m_ * x0 for m_ in m])
    intersections = [intersection_line_ellipse(m, n, e, x0, y0) for e in ellipses]
    
    # Compute cup-to-disc ratio profile
    radii = np.array([distance.cdist(centre.T, i.T) for i in intersections])
    cdr_profile = radii[0,:] / radii[1,:]
    
    # Prepare output
    out = []
    out.append([x0, y0])
    out.append(intersections[0])
    out.append(intersections[1])
    out.append(np.vstack((np.arange(0, 360, ang_step), cdr_profile)))
    out.append(ellipses)
    
    return out
