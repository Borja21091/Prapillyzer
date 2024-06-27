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
        ellipses.append(ellipse)
        
    # Sort ellipses by area, smallest (cup) to largest (disc)
    ellipses = sorted(ellipses, key=lambda x: x[1][0]*x[1][1])
    
    # Intersection line (y = mx + n) with ellipses ((x-x0)^2/A^2 + (y-y0)^2/B^2 = 1)
    m = np.array([np.tan(np.deg2rad(ang)) for ang in range(0, 180, ang_step)])
    n = np.array([ellipses[0][0][1] - m_ * ellipses[0][0][0] for m_ in m])
    intersections = [intersection_line_ellipse(m, n, e) for e in ellipses]
    
    # Compute cup-to-disc ratio profile
    centre = np.array(ellipses[0][0]).reshape(-1,1)
    radii = np.array([distance.cdist(centre.T,i.T) for i in intersections])
    cdr_profile = radii[0,:] / radii[1,:]
    
    # Prepare output
    out = []
    out.append(intersections[0])
    out.append(intersections[1])
    out.append(np.vstack((np.arange(0, 360, ang_step), cdr_profile)))
    
    return out
    
def intersection_line_ellipse(m, n, ellipse):
    x0, y0 = ellipse[0]
    A, B = ellipse[1]
    C = 1 - (x0/A)**2 - (y0/B)**2
    a = B**2 + m**2*A**2
    b = 2 * A**2*m*(n - y0) - 2*x0*B**2
    c = A**2 * (n**2 - 2*n*y0 - C*B**2)
    delta = b**2 - 4*a*c
    x = np.array([(-b + np.sqrt(delta)) / (2*a), (-b - np.sqrt(delta)) / (2*a)])
    x = x.reshape(-1, 1) # Reshape to column vector
    y = np.tile(m,2).reshape(-1,1)*x + np.tile(n,2).reshape(-1,1)
    
    return np.array([x.flatten(), y.flatten()])