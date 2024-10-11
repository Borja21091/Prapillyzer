import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from definitions import *
from loguru import logger
from definitions import RESULTS_DIR
from src.geom import get_centroid
import scipy.spatial.distance as distance
from src.utils import save_results_to_csv, level_image, rotate_image, init_results_csv, generate_results_plot
from src.masking import DeepLabV3MobileNetV2, cdr_profile, generate_masks, merge_masks, clean_segmentations

from matplotlib import pyplot as plt

def process_images_in_directory(directory: str):
    
    # Set up results CSV file
    init_results_csv()
        
    # Process images in directory
    file_list = os.listdir(directory)
    file_list.sort()
    n = len(file_list)
    for i, filename in tqdm(enumerate(file_list)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            logger.debug(f"({i + 1}/{n}) Processing {filename}")
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            process_image(img, filename)
        else:
            logger.warning(f"Skipping {filename}. File extension not supported.")

def process_image(img: Image, filename: str):
    """Process an image and save the results to a CSV file.

    Parameters:
    ----------
    
        img (Image): input image as a PIL Image object.
        filename (str): image filename.
    
    """
    # Prepare dictionary to store results
    info = {'filename': filename}
    
    # Generate masks
    try:
        masks = generate_masks(img)
        info.update({'masks': True})
        logger.success("Masks generated successfully.")
    except Exception as e:
        logger.error(f"Error generating masks - {e}")
        info.update({'masks': False})
        save_results_to_csv(info)
        return
    
    # Clean segmentations
    logger.debug("Cleaning segmentations.")
    cleaned_masks = clean_segmentations(masks)
    
    # Generate unified mask
    logger.debug("Merging masks.")
    unified_mask = merge_masks(cleaned_masks)
    
    # Level image and mask
    # img_level, fov_coord, disc_coord, ang = level_image(img, masks.get('fovea'), masks.get('disc'))
    logger.debug("Leveling image and mask.")
    img_level, fov_coord, disc_coord, ang = level_image(img, unified_mask)
    cup_coord = get_centroid(cleaned_masks.get('cup'))
    mask_level = rotate_image(ang, unified_mask)
    
    # Compute cup-to-disc ratio profile
    try:
        centre, sec_cup, sec_disc, cdr, ellipses = cdr_profile(np.max(mask_level,axis=2), ang_step=5)
        # print(sec_cup, '\n#################\n', sec_disc)
        info.update({'pcdr': True})
        logger.success("Cup-to-disc ratio profile computed successfully.")
    except Exception as e:
        logger.error(f"Error computing cup-to-disc ratio profile - {e}")
        info.update({'pcdr': False})
        save_results_to_csv(info)
        return
    
    # Account for L-R eye [N S T I N]
    if fov_coord[0] > disc_coord[0]: # Fovea is to the right of the disc (left eye)
        # Update info
        info.update({'eye': 'L'})
        # [N S T I N] -> [T S N I T]
        pcdr = cdr[1,:]
        angle = cdr[0,:]
        # Find index when angle == 180
        idx = np.where(angle == 180)[0][0]
        # Rearrange pcdr: From index to end + From start to index [T I N S T], then flip [T S N I T]
        pcdr = np.flip(np.hstack((pcdr[idx:], pcdr[:idx])))
        # Add angle
        cdr = np.vstack((angle, pcdr))
    else:
        info.update({'eye': 'R'})
        
    # Calculate disc size (area of disc ellipse) / disc-fovea distance
    disc_size = np.pi * ellipses[1][1][0] * ellipses[1][1][1]
    disc_fovea_dist = np.linalg.norm(np.array(fov_coord) - np.array(disc_coord))
    norm_disc_size = disc_size / disc_fovea_dist
    
    # Calculate vCDR and hCDR
    idx = (np.where(cdr[0,:] == 90)[0][0], 
           np.where(cdr[0,:] == 270)[0][0], 
           np.where(cdr[0,:] == 0)[0][0], 
           np.where(cdr[0,:] == 180)[0][0])
    vdCup = distance.cdist([sec_cup[:, idx[0]]], [sec_cup[:, idx[1]]])[0][0]
    vdDisc = distance.cdist([sec_disc[:,idx[0]]], [sec_disc[:,idx[1]]])[0][0]
    hdCup = distance.cdist([sec_cup[:,idx[2]]], [sec_cup[:,idx[3]]])[0][0]
    hdDisc = distance.cdist([sec_disc[:,idx[2]]], [sec_disc[:,idx[3]]])[0][0]
    vCDR = vdCup / vdDisc
    hCDR = hdCup / hdDisc
    
    # Update dictionary with results and save to CSV
    info.update({'fovea_x': fov_coord[0], 'fovea_y': fov_coord[1],
                 'disc_x': disc_coord[0], 'disc_y': disc_coord[1],
                 'cup_x': cup_coord[0], 'cup_y': cup_coord[1],
                 'd_discfov': disc_fovea_dist, 'disc_size': disc_size,
                 'cup_size': np.pi * ellipses[0][1][0] * ellipses[0][1][1],
                 'norm_disc_size': norm_disc_size,
                 'norm_cup_size': np.pi * ellipses[0][1][0] * ellipses[0][1][1] / disc_fovea_dist,
                 'rotation_angle': ang*180/np.pi,
                 'vCDR': vCDR, 'hCDR': hCDR})
    info.update({f'pcdr_{angle:d}': v for angle, v in zip(cdr[0,:].astype(int), cdr[1,:])})
    save_results_to_csv(info)
    logger.success(f"Results saved to CSV.")
    
    # # Save levelled mask
    # save_unified_mask(mask_level, ellipses, filename)
    
    # # Save cdr profile plot
    # save_pcdr_plot(cdr, filename)
    
    # # Save levelled image with ellipses
    # plot_levelled_image(img_level, centre, (sec_cup, sec_disc), ellipses, filename)
    try:
        generate_results_plot(img_level, centre, (sec_cup, sec_disc), ellipses, cdr, filename)
    except Exception as e:
        logger.error(f"Error generating results plot - {e}")
    
if __name__ == '__main__':
    
    # Set data path
    dataset = 'data'
    data_path = f'{dataset}' # f'/home/borja/OneDrive/Postdoc/Datasets/{dataset}/full-fundus' # '/media/borja/Seagate Expansion Drive/Rosetrees/Fundus2'
    
    # Log to file
    logger.add(os.path.join(LOG_DIR, f'pCDR_{dataset}' + '_{time}.log'))
    
    process_images_in_directory(data_path)