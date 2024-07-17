from superfeatures import level_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    
    # Load image
    img = Image.open(os.path.join(base_dir,'../images/136.jpg'))
    
    # Level image disc-fovea
    img_level, fov_coord, disc_coord, ang = level_image(img)
    
    # Plot of profile
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    # Plot fovea and disc
    plt.scatter(fov_coord[0], fov_coord[1], s=3, c='r')
    plt.scatter(disc_coord[0], disc_coord[1], s=3, c='b')
    # Plot line connecting fovea and disc
    x = [fov_coord[0], disc_coord[0]]
    y = [fov_coord[1], disc_coord[1]]
    plt.plot(x, y, 'k--', linewidth=0.5)
    # Add text at the top right corner with the angle
    plt.text(0.85, 0.95, f'Angle: {ang*180/np.pi:.2f}', ha='right', va='top', transform=plt.gca().transAxes, color=[1, 1, 1])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_level)
    plt.title('Levelled')
    plt.axis('off')
    
    plt.show()