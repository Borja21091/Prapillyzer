from src.masking import DeepLabV3MobileNetV2
from main import level_image, cdr_profile
from src.utils import rotate_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

if __name__ == '__main__':
        
    base_dir = os.path.dirname(__file__)
    
    # Load image
    img = Image.open(os.path.join(base_dir,'../images/136.jpg'))
    
    # Level image disc-fovea
    img_level, fov_coord, disc_coord, ang = level_image(img)
    
    # Load mask
    mask = Image.open(os.path.join(base_dir,'../masks/cup_disc/136.png'))
    mask = np.array(mask)
    
    # Apply ang to mask
    mask_level = rotate_image(ang, mask)
    
    # Compute cup-to-disc ratio profile
    centre, sec_cup, sec_disc, cdr = cdr_profile(mask_level, ang_step=5)
    
    # Plot of profile
    plt.figure()
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
    
    # Plot figure with intersection points
    plt.figure()
    for sec, color in zip([sec_cup, sec_disc], ['r', 'b']):
        x = sec[0]
        y = sec[1]
        plt.scatter(x, y, s=3, c=color)
    # Fit ellipse to cup
    cup = cv2.fitEllipse(np.argwhere(mask_level == 1))
    cup = ((cup[0][1], cup[0][0]), (cup[1][1], cup[1][0]), cup[2])
    cv2.ellipse(img_level, cup, (0, 255, 0), 2)
    # Fit ellipse to disc
    disc = cv2.fitEllipse(np.argwhere(mask_level == 2))
    disc = ((disc[0][1], disc[0][0]), (disc[1][1], disc[1][0]), disc[2])
    cv2.ellipse(img_level, disc, (0, 0, 255), 2)
    # Plot the lines passing through the centre of the cup
    for i in range(len(sec_cup[0])):
        plt.plot([sec_cup[0][i], sec_disc[0][i]], [sec_cup[1][i], sec_disc[1][i]], 'k--', linewidth=0.5)
    # Plot levelled image
    plt.imshow(img_level)
    # Plot centre
    plt.scatter(centre[0], centre[1], s=10, c='g')
    
    plt.show()