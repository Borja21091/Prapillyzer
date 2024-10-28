import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger
from definitions import *
from definitions import RESULTS_DIR
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
from src.geom import get_centroid, get_rotation, bbox_ellipse

####### LOADING AND SAVING FUNCTIONS #######

def init_results_csv(results_dir: str = RESULTS_DIR):
    """
    Initializes a results.csv file for storing results of a specific analysis.
    
    This function performs the following steps:
    
        1. Deletes the existing results.csv file if it exists.
        2. Creates a new results.csv file.
        3. Writes the header row to the file, specifying the column names.
        4. Logs a success message indicating that the CSV file has been created.
    
    Note:
    
        - The file is created in the current working directory.
        - If the results.csv file already exists, it will be deleted before creating a new one.
        
    Example usage:
    >>> init_results_csv()
    """
    # Delete results.csv if it exists
    if os.path.exists(os.path.join(results_dir, 'results.csv')):
        logger.trace("Deleting existing results.csv file.")
        os.remove(os.path.join(results_dir, 'results.csv'))
    
    # Iintialize CSV file
    with open(os.path.join(results_dir, 'results.csv'), 'x') as f:
        f.write('filename,masks,pcdr,eye,fovea_x,fovea_y,disc_x,disc_y,cup_x,cup_y,')
        f.write('d_discfov,disc_size,cup_size,norm_disc_size,norm_cup_size,rotation_angle,')
        f.write('vCDR,hCDR,')
        f.write(','.join([f'pcdr_{angle:d}' for angle in range(0, 180, 5)]) + ',\n')
        
    logger.success("CSV file results.csv created.")
    
def save_unified_mask(mask: np.ndarray, ellipses: tuple, filename: str):
    """
    Save the unified mask with ellipses to a file.
    
    Parameters:
    ----------
    
        mask (np.ndarray): The mask to be saved.
        ellipses (tuple): A tuple containing two ellipses to be drawn on the mask.
        filename (str): The name of the file to save the mask.
        
    Returns:
    -------
    
        None
    """
    mask_path = MASK_DIR
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
        
    # Prepare mask for saving
    out_mask = (mask * 255 / max(mask.flatten())).astype(np.uint8)
    # Make it RGB
    out_mask = cv2.merge([out_mask, out_mask, out_mask])
    cv2.ellipse(out_mask, ellipses[0], [0, 0, 255], 1)
    cv2.ellipse(out_mask, ellipses[1], [0, 255, 0], 1)
    
    cv2.imwrite(mask_path + '/' + filename, out_mask)

def save_results_to_csv(info: dict, results_dir: str = RESULTS_DIR):
    """
    Save the given information to a CSV file.

    Parameters:
    ----------
    
        info (dict): A dictionary containing the information to be saved.

    Returns:
    -------
    
        None
    """
    df = pd.DataFrame(info, index=[0])
    if not os.path.exists(os.path.join(results_dir, 'results.csv')):
        df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
    else:
        df.to_csv(os.path.join(results_dir, 'results.csv'), mode='a', header=False, index=False)

####### HELPER FUNCTIONS #######

def convert_to_tensor(img: Image, transform: Compose, device: torch.device) -> torch.Tensor:
    """Convert an image to a tensor and move it to the specified device."""
    return transform(img).to(device)

def scale_coordinates(coord: tuple, img_shape: tuple, mask_shape: tuple) -> tuple:
    """Scale coordinates from mask size to original image size."""
    x, y = coord
    x *= img_shape[1] / mask_shape[1]
    y *= img_shape[0] / mask_shape[0]
    return x, y

def fig2array(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array."""
    fig.canvas.draw()
    fig_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_array = fig_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig_array

####### PLOTTING FUNCTIONS #######

def generate_img_ellipse_plot(img: np.ndarray, centre: tuple, intersections: tuple, ellipses: list) -> np.ndarray:
        
    # Add cup ellipse
    cup = ellipses[0]
    cv2.ellipse(img, cup, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # Add disc ellipse
    disc = ellipses[1]
    cv2.ellipse(img, disc, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    
    # Plot centre
    int_centre = (int(centre[0]), int(centre[1]))
    cv2.circle(img, int_centre, 3, (214, 41, 0), -1)
    
    return img

def generate_pcdr_plot(cdr: list, ax) -> None:
    """
    Generate a plot of the cup-to-disc ratio profile.
    
    Parameters:
    ----------
    
        cdr (list): A list containing the cup-to-disc ratio data.
    """
    # Plot results in figure axes
    ax.plot(cdr[0,:], cdr[1,:], 'k--', linewidth=0.5)
    ax.scatter(cdr[0,:], cdr[1,:], s=3, c='k')
    # Overlay N S T I N labels on top of the X axis
    angle = [0, 90, 180, 270, 360]
    quadrant = ['N', 'S', 'T', 'I', 'N']
    for a, q in zip(angle, quadrant):
        ax.text(a, 0.025, q, fontsize=20, color='k', fontweight='bold')

def generate_results_plot(img: np.ndarray, centre: tuple, intersections: tuple, ellipses: list, cdr: list, filename: str) -> None:
    """Generate a montage of:
    
        - Original image cropped around the optic disc
        - Original image with the ellipses and intersection points overlaid cropped around the optic disc
        - pCDR plot
        
    The two original images will go on the first row, side by side. The pCDR plot will span the entire second row.
    
    Parameters:
    ----------
    
        img (np.ndarray): The original image.
        centre (tuple): The centre of the image.
        intersections (tuple): The intersection points of the ellipses.
        ellipses (list): The ellipses fitted to the cup and disc.
        filename (str): The name of the file to save the plot.
    """    
    # Create original cropped image
    int_centre = (int(centre[0]), int(centre[1]))
    bbox_disc = bbox_ellipse(ellipses[1])
    radius = int(max(bbox_disc[2], bbox_disc[3]) * 2.5 // 2)
    cropped_img = crop_image(img.copy(), int_centre, radius)
    
    # Create cropped image with ellipses and intersection points
    img_ellipses = generate_img_ellipse_plot(img, centre, intersections, ellipses)
    cropped_img_ellipses = crop_image(img_ellipses, int_centre, radius)
    
    # Generate montage
    plt.subplot(2, 2, 1)
    plt.imshow(cropped_img)
    plt.gca().set_title('Original image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cropped_img_ellipses)
    plt.gca().set_title('Cup & Disc segmentation')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    ax = plt.gca()
    generate_pcdr_plot(cdr, ax=ax)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cup-to-disc ratio')
    plt.title('Cup-to-disc ratio profile')
    plt.grid()
    plt.ylim([0, 1])
    
    # Save montage
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close('all')

def crop_image(img: np.ndarray, centre: tuple[int, int], radius: int) -> np.ndarray:
    """
    Crop an image around a given centre and radius.
    
    Parameters
    ----------
    
        img (np.ndarray): The input image.
        centre (tuple): The centre of the crop.
        radius (int): The radius of the crop.
        
    Returns:
    -------
    
        np.ndarray: The cropped image.
    """
    x, y = centre
    
    # Define crop boundaries
    x1 = max(0, x - radius)
    x2 = min(img.shape[1], x + radius)
    y1 = max(0, y - radius)
    y2 = min(img.shape[0], y + radius)
    
    return img[y1:y2, x1:x2]

####### IMAGE PROCESSING #######

def level_image(img:Image, mask:np.ndarray=None) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], float]:
    """
    Rotate a fundus image to make the line connecting the fovea and the disc horizontal.
    
    Parameters
    ----------
    
        img (Image): The input image.
        mask (np.ndarray): Unified mask of disc, cup and fovea.
        
    Returns:
    -------
    
        tuple[np.ndarray, tuple[int, int], tuple[int, int], float]:
            - Processed image
            - Centroid coordinates for the fovea
            - Centroid coordinates for the disc
            - Rotation angle in radians.
    """
    
    # Extract masks from unified mask
    mask_f = mask[:,:,-1]
    mask_d = mask[:,:,0]
    
    # Compute centroids
    x_f, y_f = get_centroid(mask_f)
    x_d, y_d = get_centroid(mask_d)
    
    # Scale to original image size
    x_f, y_f = scale_coordinates((x_f, y_f), np.array(img).shape, mask_f.shape)
    x_d, y_d = scale_coordinates((x_d, y_d), np.array(img).shape, mask_d.shape)
    
    # Compute rotation angle
    ang = get_rotation((x_f, y_f), (x_d, y_d), radians=True) # Radians
    
    # Rotate image
    out_img = rotate_image(ang, np.array(img))
    
    return out_img, (x_f, y_f), (x_d, y_d), ang

def rotate_2d_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D image by a given angle."""
    rot_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), np.rad2deg(angle), 1)
    img = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
    return img

def rotate_image(ang: float, img: np.ndarray) -> np.ndarray:
    """
    Rotate a 2D or 3D image by a given angle.
    
    Parameters
    ----------
    ang : float
        Rotation angle in radians.
        
    img : np.ndarray
        Input image as a np.ndarray object.
        
    Returns
    -------
    np.ndarray
        Rotated image.
    """
    if img.ndim == 2:
        return rotate_2d_image(img, ang)
    elif img.ndim >= 3:
        rotated_slices = [rotate_2d_image(img[..., i], ang) for i in range(img.shape[2])]
        return np.stack(rotated_slices, axis=2)
    else:
        raise ValueError("Input image must be at least 2D.")
