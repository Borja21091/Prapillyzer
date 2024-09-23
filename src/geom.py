import cv2
import numpy as np

def is_ellipse_contained(inner:cv2.RotatedRect, outter:cv2.RotatedRect) -> bool:
    """
    Check if the inner ellipse is completely contained within the outer ellipse.
    
    Parameters:
    ----------
    
        inner (cv2.RotatedRect): The inner ellipse represented as a RotatedRect object.
        outter (cv2.RotatedRect): The outer ellipse represented as a RotatedRect object.
        
    Returns:
    -------
    
        bool: True if the inner ellipse is completely contained within the outer ellipse, False otherwise.
    """
    
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

def implicit_ellipse(ellipse) -> tuple:
    """
    Calculates the coefficients of the implicit equation of an ellipse.
    
    Parameters:
    ----------
        
        ellipse (tuple): A tuple representing the ellipse, containing the following elements:
            - ellipse[0] (tuple): The center coordinates of the ellipse (x0, y0).
            - ellipse[1] (tuple): The major and minor axes of the ellipse (a, b).
            - ellipse[2] (float): The angle of rotation of the ellipse in degrees.
        
    Returns:
    -------
    
        tuple: A tuple containing the coefficients of the implicit equation of the ellipse (A, B, C, D, E, F).
    """
    
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

def intersection_line_ellipse(m, n, ellipse, x0, y0) -> np.ndarray:
    """
    Calculates the intersection points between a line and an ellipse.
    
    Parameters:
    ----------
    
        m (float): Slope of the line.
        n (float): Y-intercept of the line.
        ellipse (tuple): Tuple containing the coefficients of the ellipse equation (A, B, C, D, E, F).
        x0 (float): X-coordinate of the center of the ellipse.
        y0 (float): Y-coordinate of the center of the ellipse.
        
    Returns:
    -------
    
        np.ndarray: Array containing the x and y coordinates of the intersection points.
    """
    
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

def get_centroid(mask:np.ndarray) -> tuple:
    """
    Compute the centroid of a binary mask.
    
    Parameters
    ----------
    
    mask (np.ndarray): Mask as a np.ndarray object.
        
    Returns
    -------
        tuple: Tuple containing the centroid coordinates (x, y).
    """
    # Binarize mask
    mask = mask / max(mask.flatten())
    # Compute centroid
    x_centroid = np.sum(np.argwhere(mask)[:,1]) / np.sum(mask)
    y_centroid = np.sum(np.argwhere(mask)[:,0]) / np.sum(mask)
    
    return x_centroid, y_centroid

def get_rotation(centroid_fovea:tuple, centroid_disc:tuple, radians:bool=True) -> float:
    """
    Calculate rotation angle between two points.

    Parameters:
    ----------
    
        centroid_fovea (tuple): (x, y) coordinates of the fovea centroid.
        centroid_disc (tuple): (x, y) coordinates of the disc centroid.
        radians (bool, optional): Radians/Degrees of output. Defaults to radians.
        
    Returns:
    -------
        float: Rotation angle in radians or degrees.
    """
    # Compute rotation angle
    # rotation = np.arctan2(centroid_disc[1] - centroid_fovea[1], centroid_disc[0] - centroid_fovea[0])
    rotation = np.arctan((centroid_disc[1] - centroid_fovea[1]) / (centroid_disc[0] - centroid_fovea[0]))
    
    # Convert to degrees
    if not radians:
        rotation = np.rad2deg(rotation)
    
    return rotation
