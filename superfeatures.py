import cv2
import numpy as np
import scipy.spatial.distance as distance

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
    assert np.unique(mask).size == 3, "Mask must contain 3 unique pixel values, 0 being 'Background'."
    
    # Check angular step
    assert ang_step >= 1, "Angular step must be greater than or equal to 1."
    assert ang_step <= 180, "Angular step must be less than or equal to 180."
    assert np.mod(180, ang_step) == 0, "180 must be divisible by the angular step."
    
    # Compute ellipses for each unique pixel value except 0
    ellipses = []
    for i in np.unique(mask)[1:]:
        ellipse = cv2.fitEllipse(np.argwhere(mask == i))
        ellipse = ((ellipse[0][1], ellipse[0][0]), (ellipse[1][1], ellipse[1][0]), ellipse[2])
        ellipses.append(ellipse)
        
    # Sort ellipses by area, smallest (cup) to largest (disc)
    ellipses = sorted(ellipses, key=lambda x: x[1][0]*x[1][1])
    x0, y0 = ellipses[0][0]
    
    # Intersection line with ellipses
    m = np.array([np.tan(np.deg2rad(ang)) for ang in np.arange(0, 180, ang_step)])
    n = np.array([ellipses[0][0][1] - m_ * ellipses[0][0][0] for m_ in m])
    intersections = [intersection_line_ellipse(m, n, e, x0, y0) for e in ellipses]
    
    # Compute cup-to-disc ratio profile
    centre = np.array(ellipses[0][0]).reshape(-1,1)
    radii = np.array([distance.cdist(centre.T, i.T) for i in intersections])
    cdr_profile = radii[0,:] / radii[1,:]   
    
    # Prepare output
    out = []
    out.append(intersections[0])
    out.append(intersections[1])
    out.append(np.vstack((np.arange(0, 360, ang_step), cdr_profile)))
    
    return out
    
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