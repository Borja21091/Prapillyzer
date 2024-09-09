import os
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

def init_results_csv():
    
    # Delete results.csv if it exists
    if os.path.exists('results.csv'):
        logger.trace("Deleting existing results.csv file.")
        os.remove('results.csv')
    
    # Iintialize CSV file
    with open('results.csv', 'x') as f:
        f.write('filename,masks,pcdr,eye,fovea_x,fovea_y,disc_x,disc_y,cup_x,cup_y,')
        f.write('d_discfov,disc_size,cup_size,norm_disc_size,norm_cup_size,rotation_angle,')
        f.write('vCDR,hCDR,')
        f.write(','.join([f'pcdr_{angle:d}' for angle in range(0, 180, 5)]) + ',\n')
        
    logger.success("CSV file results.csv created.")

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

def intersection_line_ellipse(m, n, ellipse, x0, y0):
    
    A, B, C, D, E, F = implicit_ellipse(ellipse)
    
    # Intersection points
    a = A + m*(B + m*C)
    b = D + m*(2*C*n + E) + B*n
    c = n*(C*n + E) + F
    delta = b**2 - 4*a*c
    x = np.array([(-b + np.sqrt(delta)) / (2*a), (-b - np.sqrt(delta)) / (2*a)])
    x = x.reshape(-1, 1) # Reshape to column vector
    y = np.tile(m,2).reshape(-1,1)*x + np.tile(n,2).reshape(-1,1)
    
    # Angle subdivision
    ang_step = 360 / len(y)
    
    # Find index of vertical intersections (90, 270 degrees)
    idx = (np.array([90, 270]) / ang_step).astype(int)
    
    # Handle vertical intersections
    y_ = np.roots([C, B*x0 + E, A*x0**2 + D*x0 + F])
    y_ = np.sort(y_, axis=0)[::-1].reshape(-1,1) # Sort in descending order
    
    # Replace with vertical intersections
    x[idx,0] = x0
    y[idx] = y_
    
    # Prepare output
    out = np.array([x.flatten(), y.flatten()])
    # Sort circularly from 0 to 360 degrees
    ang = np.arctan2(out[1,:] - y0, out[0,:] - x0) % (2*np.pi)
    idx = np.argsort(ang)[::-1]
    out = out[:,idx]
    
    return out
    
def implicit_ellipse(ellipse):
    
    x0, y0 = ellipse[0]
    a, b = tuple(ti/2 for ti in ellipse[1])
    theta = np.deg2rad(ellipse[2])
    
    A = a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2
    B = 2*(b**2 - a**2)*np.sin(theta)*np.cos(theta)
    C = a**2*np.cos(theta)**2 + b**2*np.sin(theta)**2
    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
    
    return A, B, C, D, E, F

def is_ellipse_contained(inner:cv2.RotatedRect, outter:cv2.RotatedRect) -> bool:
    
    # Templates of zeros (512 x 512)
    _cup = np.zeros((512, 512), np.uint8)
    _disc = np.zeros((512, 512), np.uint8)
    
    # Draw ellipses
    cv2.ellipse(_cup, inner, 1, -1)
    cv2.ellipse(_disc, outter, 1, -1)
    
    # Logical AND
    _and = np.logical_and(_cup, _disc)
    
    # Check if cup == cup AND disc
    return np.array_equal(_cup, _and)
    
def get_centroid(mask:np.ndarray)->tuple:
    """
    Compute the centroid of a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask as a np.ndarray object.
        
    Returns
    -------
    tuple
        Tuple containing the centroid coordinates (x, y).
    """    
    # Compute centroid
    x_centroid = np.sum(np.argwhere(mask)[:,1]) / np.sum(mask)
    y_centroid = np.sum(np.argwhere(mask)[:,0]) / np.sum(mask)
    
    return x_centroid, y_centroid

def get_rotation(centroid_fovea:tuple, centroid_disc:tuple, radians:bool=True)->float:
    """
    Calculate rotation angle between two points.

    Args:
        centroid_fovea (tuple): (x, y) coordinates of the fovea centroid.
        centroid_disc (tuple): (x, y) coordinates of the disc centroid.
        radians (bool, optional): Radians/Degrees of output. Defaults to radians.
        
    Returns:
        float: Rotation angle in radians or degrees.
    """
    # Compute rotation angle
    # rotation = np.arctan2(centroid_disc[1] - centroid_fovea[1], centroid_disc[0] - centroid_fovea[0])
    rotation = np.arctan((centroid_disc[1] - centroid_fovea[1]) / (centroid_disc[0] - centroid_fovea[0]))
    
    # Convert to degrees
    if not radians:
        rotation = np.rad2deg(rotation)
    
    return rotation

def rotate_image(ang:float, img:np.ndarray)->np.ndarray:
    """
    Rotate an image by a given angle.
    
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
    
    # Rotation matrix
    rot_matrix = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), np.rad2deg(ang), 1)
    
    # Rotate image 'ang' radians
    img = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
    
    return img