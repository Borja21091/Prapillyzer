import os
import torch

def mask_fovea(img:torch.Tensor) -> torch.Tensor:
    """
    Fovea masking of a fundus image.

    Args:
        img (torch.Tensor): input RGB image as a torch.Tensor object.

    Returns:
        torch.Tensor: tensor with the fovea masked. 0 values are considered background.    
    """
    # Load pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 'models/fovea.pth')
    model = torch.load(model_path)
    
    # Mask fovea
    img = img.unsqueeze(0)
    mask = model(img)
    mask = torch.argmax(mask, dim=1).squeeze(0)
    
    return mask

def mask_disc(img:torch.Tensor) -> torch.Tensor:
    """
    Disc masking of a fundus image.

    Args:
        img (torch.Tensor): input RGB image as a torch.Tensor object.

    Returns:
        torch.Tensor: tensor with the disc masked. 0 values are considered background.    
    """
    # Load pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 'models/disc.pth')
    model = torch.load(model_path)
    
    # Mask disc
    img = img.unsqueeze(0)
    mask = model(img)
    mask = torch.argmax(mask, dim=1).squeeze(0)
    
    return mask