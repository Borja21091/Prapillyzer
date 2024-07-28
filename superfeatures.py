import os
import cv2
import torch
import numpy as np
from PIL import Image
from src.utils import *
import scipy.spatial.distance as distance
from src.masking import DeepLabV3MobileNetV2
from src.masking import mask_fovea, mask_disc
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def process_images_in_directory(directory: str):
    
    # Delete results.csv if it exists
    if os.path.exists('results.csv'):
        os.remove('results.csv')
        
    # Process images in directory
    file_list = os.listdir(directory)
    file_list.sort()
    n = len(file_list)
    for i, filename in enumerate(file_list):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"({i + 1}/{n}) Processing {filename}")
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
    info = {'filename': filename}
    
    # Generate masks
    try:
        masks = generate_masks(img)
        info.update({'masks': True})
    except:
        print(f"Error generating masks for {filename}.")
        info.update({'masks': False})
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
    try:
        centre, sec_cup, sec_disc, cdr, ellipses = cdr_profile(mask_level, ang_step=5)
        info.update({'pcdr': True})
    except:
        print(f"Error computing cup-to-disc ratio profile for {filename}.")
        info.update({'pcdr': False})
        save_results_to_csv(info)
        return
    
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
    else:
        info.update({'eye': 'R'})
    
    # Update dictionary with results and save to CSV
    info.update({'fovea_x': fov_coord[0], 'fovea_y': fov_coord[1],
                 'disc_x': disc_coord[0], 'disc_y': disc_coord[1],
                 'cup_x': cup_coord[0], 'cup_y': cup_coord[1],
                 'rotation_angle': ang*180/np.pi})
    info.update({f'pcdr_{angle:d}': v for angle, v in zip(cdr[0,:].astype(int), cdr[1,:])})
    save_results_to_csv(info)
    
    # # Save levelled mask
    # save_unified_mask(mask_level, ellipses, filename)
    
    # # Save cdr profile plot
    # save_pcdr_plot(cdr, filename)
    
    # # Save levelled image with ellipses
    # plot_levelled_image(img_level, centre, (sec_cup, sec_disc), ellipses, filename)
    
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
    data_path = 'data' # '/home/borja/OneDrive/Postdoc/Datasets/SMDG/full-fundus'
    process_images_in_directory(data_path)