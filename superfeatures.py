import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import scipy.spatial.distance as distance
from src.masking import DeepLabV3MobileNetV2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops_table
from src.masking import mask_fovea, mask_disc, mask_cup
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from src.utils import intersection_line_ellipse, get_centroid, get_rotation, rotate_image, fit_ellipse

def process_images_in_directory(directory: str):
    # Delete results.csv if it exists
    if os.path.exists('results.csv'):
        os.remove('results.csv')
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"Processing {filename}")
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            process_image(img, filename)

def process_image(img: Image, filename: str):
    info = {'filename': filename}
    masks = generate_masks(img)
    if not masks:
        print(f"Areas not detected in {filename}")
        info.update({'pcdr': None})
        save_results_to_csv(info)
        return

    cleaned_masks = clean_segmentations(masks)    
    ellipses = fit_ellipse(cleaned_masks)
    if not ellipses:
        print(f"Ellipses not properly contained in {filename}")
        info.update({'pcdr': None})
        save_results_to_csv(info)
        return

    unified_mask = merge_masks(cleaned_masks)
    save_unified_mask(unified_mask, filename)
    img_level, fov_coord, disc_coord, ang = level_image(img, masks.get('fovea'), masks.get('disc'))
    mask_level = rotate_image(ang, unified_mask)
    _, _, _, cdr = cdr_profile(mask_level.astype(np.uint8), ang_step=5)
    info.update({f'pcdr_{angle:d}': v for angle, v in zip(cdr[0,:].astype(int), cdr[1,:])})

    save_results_to_csv(info)

def save_unified_mask(mask: np.ndarray, filename: str):
    mask_path = './data/masks'
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    cv2.imwrite(mask_path + '/' + filename, (mask * 255 / max(mask.flatten())).astype(np.uint8))

def generate_masks(img: Image) -> dict:
    # Convert image to tensor
    transform_disc = Compose([Resize((512, 512)), ToTensor(), Normalize((0.5,), (0.5,))])
    transform_fovea = Compose([Resize((224, 224)), ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img4fovea= transform_fovea(img).to(device)
    img4disc_cup = transform_disc(img).to(device)
    
    # Process image
    mask_d = mask_disc(img4disc_cup)
    mask_c = mask_cup(img4disc_cup)
    mask_f = mask_fovea(img4fovea)
    
    # Prepare output
    keys = ['disc', 'cup', 'fovea']
    values = [mask_d, mask_c, mask_f]
    output = {k: v.to(torch.uint8).cpu().numpy() if sum(v.flatten() != 0) else None for k, v in zip(keys, values)}
    
    return output

def clean_segmentations(masks: dict) -> dict:
    cleaned_masks = {}
    for key, mask in masks.items():
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cleaned_mask_labelled = label(cleaned_mask)
        props = regionprops_table(cleaned_mask_labelled, properties=('area', 'perimeter'))
        radii = props.get('perimeter') / (2 * np.pi) + 0.5
        roundness = roundness = [(4 * np.pi * props.get('area')[idx] / (props.get('perimeter')[idx] ** 2)) * (1 - 0.5 * r)**2 
                                if props.get('perimeter')[idx] != 0 else 0.0 for idx, r in enumerate(radii)]
        if len(roundness) != 0:
            # Remove the ones with an area smaller than 50 px
            roundness = np.array(roundness)
            roundness[props.get('area') < 50] = 0
            idx = np.argmax(roundness)
            cleaned_mask = (cleaned_mask_labelled == idx + 1)
        cleaned_masks[key] = cleaned_mask.astype(np.uint8)
    return cleaned_masks

def merge_masks(masks: dict) -> np.ndarray:
    unified_mask = np.zeros_like(next(iter(masks.values()))).astype(float)
    for idx, mask in enumerate(masks.values()):
        unified_mask[mask.astype(bool)] = idx + 1
    return unified_mask

def save_results_to_csv(info:dict):
    df = pd.DataFrame(info, index=[0])
    if not os.path.exists('results.csv'):
        df.to_csv('results.csv', index=False)
    else:
        df.to_csv('results.csv', mode='a', header=False, index=False)

def level_image(img:Image, mask_f:np.ndarray=None, mask_d:np.ndarray=None) -> np.ndarray:
    """
    Rotate a fundus image to make the line connecting the fovea and the disc horizontal.
    
    Parameters
    ----------
    img : np.ndarray
        RGB fundus image.
    
    Returns
    -------
    np.ndarray
        Straightened RGB fundus image.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert image to tensor
    transform_disc = Compose([Resize((512, 512)), ToTensor(), Normalize((0.5,), (0.5,))])
    transform_fovea = Compose([Resize((224, 224)), ToTensor()])
    
    # Convert to tensors
    img4fovea= transform_fovea(img).to(device)
    img4disc = transform_disc(img).to(device)
    
    # Mask fovea and disc
    if mask_f is None:
        mask_f = mask_fovea(img4fovea).cpu().numpy()

    if mask_d is None:
        mask_d = mask_disc(img4disc).cpu().numpy()
    
    # Compute centroids
    x_f, y_f = get_centroid(mask_f)
    x_d, y_d = get_centroid(mask_d)
    
    # Scale to original image size
    x_f *= np.array(img).shape[1]/224
    y_f *= np.array(img).shape[0]/224
    x_d *= np.array(img).shape[1]/512
    y_d *= np.array(img).shape[0]/512
    
    # Compute rotation angle
    ang = get_rotation((x_f, y_f), (x_d, y_d), radians=True) # Radians
    
    # Rotate image
    out_img = rotate_image(ang, np.array(img))
    
    return out_img, (x_f, y_f), (x_d, y_d), ang
    
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
    assert mask.dtype == np.uint8, "Mask must be a greyscale image."
    assert np.unique(mask).size >= 3, "Mask must contain at least 3 unique pixel values, 0 being 'Background', 1 'Disc', 2 'Cup'."
    
    # Check angular step
    assert ang_step >= 1, "Angular step must be greater than or equal to 1."
    assert ang_step <= 180, "Angular step must be less than or equal to 180."
    assert np.mod(180, ang_step) == 0, "180 must be divisible by the angular step."
    
    # Compute ellipses for pixel values 1 and 2 (disc and cup)
    ellipses = []
    for i in [1, 2]:
        ellipse = cv2.fitEllipse(np.argwhere(mask == i))
        ellipse = ((ellipse[0][1], ellipse[0][0]), (ellipse[1][1], ellipse[1][0]), ellipse[2])
        ellipses.append(ellipse)
        
    # Sort ellipses by area, smallest (cup) to largest (disc)
    ellipses = sorted(ellipses, key=lambda x: x[1][0]*x[1][1])
    
    # Calculate centre as the midpoint between cup and disc ellipses centres
    centre = ((np.array(ellipses[0][0]) + np.array(ellipses[1][0])) / 2).reshape(-1,1)
    x0, y0 = centre[:,0]
    
    # Intersection line with ellipses
    m = np.array([np.tan(np.deg2rad(ang)) for ang in np.arange(0, 180, ang_step)])
    n = np.array([y0 - m_ * x0 for m_ in m])
    intersections = [intersection_line_ellipse(m, n, e, x0, y0) for e in ellipses]
    print(intersections)
    # Compute cup-to-disc ratio profile
    radii = np.array([distance.cdist(centre.T, i.T) for i in intersections])
    cdr_profile = radii[0,:] / radii[1,:]   
    
    # Prepare output
    out = []
    out.append([x0, y0])
    out.append(intersections[0])
    out.append(intersections[1])
    out.append(np.vstack((np.arange(0, 360, ang_step), cdr_profile)))
    
    return out

if __name__ == '__main__':
    process_images_in_directory('data')