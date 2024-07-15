import cv2
import numpy as np
import scipy.spatial.distance as distance
from src.utils import intersection_line_ellipse

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