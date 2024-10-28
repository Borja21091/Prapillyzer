import os
import configparser

# Read configuration file
config = configparser.ConfigParser()
config.read("cfg/config.ini")

##################### PATH DEFINITIONS #####################

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = config["PATH"]["in"]

# Results directory
RESULTS_DIR = config["PATH"]["out"]

# Mask directory
MASK_DIR = os.path.join(ROOT_DIR, 'masks')

# Models directory
MODELS_DIR = os.path.join(ROOT_DIR, 'src', 'models')

# Log directory
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

##################### FLAG DEFINITIONS #####################

# Save result image
SAVE_RESULT_IMG = config.getboolean("FLAG", "result_image")