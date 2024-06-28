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
    sec_cup, sec_disc, cdr = cdr_profile(mask, ang_step=5)
    
    # Plot of profile
    plt.figure()
    plt.plot(cdr[0,:], cdr[1,:], 'k--', linewidth=0.5)
    plt.scatter(cdr[0,:], cdr[1,:], s=3, c='k')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cup-to-disc ratio')
    plt.title('Cup-to-disc ratio profile')
    plt.ylim([0, 1])
    plt.grid()
    
    # Plot figure with intersection points
    plt.figure()
    for sec, color in zip([sec_cup, sec_disc], ['r', 'b']):
        x = sec[0]
        y = sec[1]
        plt.plot(x, y, color + 'o', markersize=2)
    # Fit ellipse to cup
    cup = cv2.fitEllipse(np.argwhere(mask == 1))
    cup = ((cup[0][1], cup[0][0]), (cup[1][1], cup[1][0]), cup[2])
    cv2.ellipse(img, cup, (0, 255, 0), 2)
    # Fit ellipse to disc
    disc = cv2.fitEllipse(np.argwhere(mask == 2))
    disc = ((disc[0][1], disc[0][0]), (disc[1][1], disc[1][0]), disc[2])
    cv2.ellipse(img, disc, (0, 0, 255), 2)
    # Plot the lines passing through the centre of the cup
    for i in range(len(sec_cup[0])):
        plt.plot([sec_cup[0][i], sec_disc[0][i]], [sec_cup[1][i], sec_disc[1][i]], 'k--', linewidth=0.5)
    plt.imshow(img)
    
    plt.show()