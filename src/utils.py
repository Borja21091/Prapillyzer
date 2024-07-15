import torch
import numpy as np
    
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
    
    # Replace zero values with nan (vertical intersections)
    y[np.isclose(y, 0)] = np.nan
    
    # Handle vertical intersections
    y_ = np.roots([C, B*x0 + E, A*x0**2 + D*x0 + F])
    y_ = np.sort(y_, axis=0)[::-1] # Sort in descending order
    
    # Replace nan values with vertical intersections
    x[np.isnan(y)] = x0
    y[np.isnan(y)] = y_
    
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

def get_centroid(mask:torch.Tensor)->tuple:
    """
    Compute the centroid of a binary mask.
    
    Parameters
    ----------
    mask : torch.Tensor
        Binary mask as a torch.Tensor object.
        
    Returns
    -------
    tuple
        Tuple containing the centroid coordinates (x, y).
    """
    # Check mask
    assert mask.ndim == 2, "Mask must be a 2D array."
    assert mask.dtype == torch.uint8, "Mask must be a binary image."
    assert torch.unique(mask).size(0) == 2, "Mask must contain 2 unique pixel values."
    
    # Compute centroid
    y, x = torch.where(mask == 1)
    x = x.float()
    y = y.float()
    x_centroid = torch.mean(x)
    y_centroid = torch.mean(y)
    
    return x_centroid, y_centroid

def get_rotation(centroid:tuple, radians:bool=True)->float:
    """
    Compute the rotation angle of a binary mask.
    
    Parameters
    ----------
    centroid : tuple
        Tuple containing the centroid coordinates (x, y).
        
    radians : bool
        If True, the rotation angle is returned in radians. If False, the rotation angle is returned in degrees.
        
    Returns
    -------
    float
        Rotation angle in degrees.
    """
    # Check centroid
    assert isinstance(centroid, tuple), "Centroid must be a tuple."
    assert len(centroid) == 2, "Centroid must contain two elements."
    
    # Compute rotation
    x, y = centroid
    rotation = torch.atan2(y, x)
    
    # Convert to degrees
    if not radians:
        rotation = torch.rad2deg(rotation)
    
    return rotation