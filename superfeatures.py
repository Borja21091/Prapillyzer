import cv2
import torch
import numpy as np
import scipy.spatial.distance as distance
from src.masking import mask_fovea, mask_disc
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from src.utils import intersection_line_ellipse, get_centroid, get_rotation, rotate_image

def level_image(img) -> np.ndarray:
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
    mask_f = mask_fovea(img4fovea)
    mask_d = mask_disc(img4disc)
    
    # Compute centroids
    x_f, y_f = get_centroid(mask_f.cpu().numpy())
    x_d, y_d = get_centroid(mask_d.cpu().numpy())
    
    # Scale to original image size
    x_f *= np.array(img).shape[1]/224
    y_f *= np.array(img).shape[0]/224
    x_d *= np.array(img).shape[1]/512
    y_d *= np.array(img).shape[0]/512
    
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