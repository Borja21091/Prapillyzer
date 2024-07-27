import cv2
import numpy as np
from scipy.optimize import fsolve
    
def intersection_line_ellipse(m, n, ellipse, x0, y0):
    
    A, B, C, D, E, F = implicit_ellipse(ellipse)
    
    # Intersection points
    a = A + m*(B + m*C)
    b = D + m*(2*C*n + E) + B*n
    c = n*(C*n + E) + F
    delta = b**2 - 4*a*c
    x = np.array([(-b + np.sqrt(delta)) / (2*a), (-b - np.sqrt(delta)) / (2*a)])
    x = x.reshape(-1, 1) # Reshape to column vector
    # x = np.vstack((x[0,:].T, x[1,::-1].T)).reshape(-1, 1)
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

def fit_ellipse(masks: dict) -> dict:
    ellipses = {}
    keys = ['cup', 'disc']
    for key in keys:
        mask = masks.get(key)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        ellipse = cv2.fitEllipse(contours[0])
        ellipses[key] = ellipse
    if not('cup' in ellipses) or not('disc' in ellipses):
        return None
    if not is_ellipse_contained(ellipses['cup'], ellipses['disc']):
        return None
    return ellipses

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
    rotation = np.arctan2(centroid_disc[1] - centroid_fovea[1], centroid_disc[0] - centroid_fovea[0])
    
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