from superfeatures.superfeatures import cdr_profile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    
    # Load image
    img = Image.open(os.path.join(base_dir,'../images/136.jpg'))
    img = np.array(img)
    
    # Load mask
    mask = Image.open(os.path.join(base_dir,'../masks/cup_disc/136.png'))
    mask = np.array(mask)
    
    # Compute cup-to-disc ratio profile
    sec_cup, sec_disc, cdr = cdr_profile(mask)
    
    # Plot of profile
    plt.figure()
    plt.plot(cdr[0,:], cdr[1,:])
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cup-to-disc ratio')
    plt.title('Cup-to-disc ratio profile')
    plt.grid()
    
    # Plot figure with intersection points
    plt.figure()
    for sec, color in zip([sec_cup, sec_disc], ['r', 'b']):
        x = sec[0]
        y = sec[1]
        plt.plot(y, x, color + 'o')
    plt.imshow(img)
    
    plt.show()