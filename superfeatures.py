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
from src.utils import intersection_line_ellipse, get_centroid, get_rotation, rotate_image, is_ellipse_contained

def process_images_in_directory(directory: str):
    
    # Delete results.csv if it exists
    if os.path.exists('results.csv'):
        os.remove('results.csv')
        
    # Process images in directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"Processing {filename}")
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            process_image(img, filename)

def process_image(img: Image, filename: str):
    """Process an image and save the results to a CSV file.

    Args:
        img (Image): input image as a PIL Image object.
        filename (str): image filename.
    """
    # Prepare dictionary to store results
    info = {'filename': filename, 
            'eye': 'R'}
    
    # Generate masks
    masks = generate_masks(img)
    if not masks:
        print(f"Areas not detected in {filename}")
        info.update({'pcdr': None})
        save_results_to_csv(info)
        return
    
    # Clean segmentations
    cleaned_masks = clean_segmentations(masks)
    
    # Generate unified mask and save it
    unified_mask = merge_masks(cleaned_masks)
    
    # Level image and mask
    img_level, fov_coord, disc_coord, ang = level_image(img, masks.get('fovea'), masks.get('disc'))
    cup_coord = get_centroid(cleaned_masks.get('cup'))
    mask_level = rotate_image(ang, unified_mask)
    
    # Compute cup-to-disc ratio profile
    centre, sec_cup, sec_disc, cdr, ellipses = cdr_profile(mask_level, ang_step=5)
    
    # Account for L-R eye [N S T I N]
    if fov_coord[0] > disc_coord[0]: # Fovea is to the right of the disc (left eye)
        # Update info
        info.update({'eye': 'L'})
        # [N S T I N] -> [T S N I T]
        pcdr = cdr[1,:]
        angle = cdr[0,:]
        # Find index when angle == 180
        idx = np.where(angle == 180)[0][0]
        # Rearrange pcdr: From index to end + From start to index [T I N S T], then flip [T S N I T]
        pcdr = np.flip(np.hstack((pcdr[idx:], pcdr[:idx])))
        # Add angle
        cdr = np.vstack((angle, pcdr))
    
    # Save levelled mask
    save_unified_mask(mask_level, ellipses, filename)
    
    # Update dictionary with results and save to CSV
    info.update({'fovea_x': fov_coord[0], 'fovea_y': fov_coord[1],
                 'disc_x': disc_coord[0], 'disc_y': disc_coord[1],
                 'cup_x': cup_coord[0], 'cup_y': cup_coord[1],
                 'rotation_angle': ang*180/np.pi})
    info.update({f'pcdr_{angle:d}': v for angle, v in zip(cdr[0,:].astype(int), cdr[1,:])})
    save_results_to_csv(info)
    
    # Save cdr profile plot
    save_pcdr_plot(cdr, filename)
    
    # Save levelled image with ellipses
    # plot_levelled_image(img_level, centre, (sec_cup, sec_disc), ellipses, filename)
    
def plot_levelled_image(img: np.ndarray, centre: tuple, intersections: tuple, ellipses: list, filename: str):
    
    plt.figure(figsize=(5.12, 5.12))
    
    # Plot intersection points
    sec_cup, sec_disc = intersections
    for sec, color in zip([sec_cup, sec_disc], ['r', 'b']):
        x = sec[0]
        y = sec[1]
        plt.scatter(x, y, s=3, c=color)
        
    # Add cup ellipse
    cup = ellipses[0]
    cv2.ellipse(img, cup, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # Add disc ellipse
    disc = ellipses[1]
    cv2.ellipse(img, disc, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    
    # Plot the lines joining the intersection points
    for i in range(len(sec_cup[0])):
        plt.plot([sec_cup[0][i], sec_disc[0][i]], [sec_cup[1][i], sec_disc[1][i]], 'k--', linewidth=0.5)
        
    # Show levelled image
    plt.imshow(img)
    
    # Plot centre
    plt.scatter(centre[0], centre[1], s=10, c='g')
    
    # Add title
    plt.title(filename.split('.')[0])
    
    plt.show()

def save_pcdr_plot(cdr: list, filename: str):
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(cdr[0,:], cdr[1,:], 'k--', linewidth=0.5)
    plt.scatter(cdr[0,:], cdr[1,:], s=3, c='k')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cup-to-disc ratio')
    plt.title('Cup-to-disc ratio profile')
    plt.ylim([0, 1])
    plt.grid()
    # Overlay N S T I N labels on top of the X axis
    angle = [0, 90, 180, 270, 360]
    quadrant = ['N', 'S', 'T', 'I', 'N']
    for a, q in zip(angle, quadrant):
        plt.text(a, 0.025, q, fontsize=20, color='k', fontweight='bold')
        
    # Save plot
    plot_path = './data/pcdr_plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.savefig(plot_path + '/' + filename.split('.')[0] + '.png')
    plt.close()
    
def save_unified_mask(mask: np.ndarray, ellipses: tuple, filename: str):
    """Save a mask to a file.

    Args:
        mask (np.ndarray): mask as a numpy array. 0 values are considered background.
        filename (str): mask filename.
    """
    mask_path = './data/masks'
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
        
    # Prepare mask for saving
    out_mask = (mask * 255 / max(mask.flatten())).astype(np.uint8)
    # Make it RGB
    out_mask = cv2.merge([out_mask, out_mask, out_mask])
    cv2.ellipse(out_mask, ellipses[0], [0, 0, 255], 1)
    cv2.ellipse(out_mask, ellipses[1], [0, 255, 0], 1)
    
    cv2.imwrite(mask_path + '/' + filename, out_mask)

def save_results_to_csv(info:dict):
    df = pd.DataFrame(info, index=[0])
    if not os.path.exists('results.csv'):
        df.to_csv('results.csv', index=False)
    else:
        df.to_csv('results.csv', mode='a', header=False, index=False)

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
    mask_d = mask_disc(img4disc_cup)
    mask_c = mask_cup(img4disc_cup)
    mask_f = mask_fovea(img4fovea)
    
    # Prepare output
    keys = ['disc', 'cup', 'fovea']
    values = [mask_d, mask_c, mask_f]
    output = {k: v.to(torch.uint8).cpu().numpy() if sum(v.flatten() != 0) else None for k, v in zip(keys, values)}
    
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
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Label and compute roundness
        cleaned_mask_labelled = label(cleaned_mask)
        props = regionprops_table(cleaned_mask_labelled, properties=('area', 'perimeter'))
        radii = props.get('perimeter') / (2 * np.pi) + 0.5
        roundness = roundness = [(4 * np.pi * props.get('area')[idx] / (props.get('perimeter')[idx] ** 2)) * (1 - 0.5 * r)**2 
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
    x_f *= np.array(img).shape[1]/mask_f.shape[1]
    y_f *= np.array(img).shape[0]/mask_f.shape[0]
    x_d *= np.array(img).shape[1]/mask_d.shape[1]
    y_d *= np.array(img).shape[0]/mask_d.shape[0]
    
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
        raise ValueError('Cup ellipse is not contained within disc ellipse.')
    
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

if __name__ == '__main__':
    process_images_in_directory('data')