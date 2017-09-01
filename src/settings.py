"""
Global settings for the project

"""
# User defined settings
IMAGES_FOLDER_NAME = 'ALL_SET_2'
IMAGE_TYPE = "jpeg"
MODEL_FILE_NAME = "saves/SAVE_2017-06-01_042925_400"
TRAINING_DATASET_FOLDER = 'synthetic'
SYNTHETIC_DATASET_FOLDER = 'synthetic'
# Set the training dataset folder to use the synthetic dataset folder if
# you want to train using synthetically generated images
# TRAINING_DATASET_FOLDER = SYNTHETIC_DATASET_FOLDER

# Directory paths are relative to the src folder
DATA_FOLDER = '../data/'
CACHE_FOLDER = '../cache/'
OUTPUT_FOLDER = '../output/'
RESULTS_FOLDER = '../results/'
MODELS_FOLDER = '../models/'
TRAINING_FOLDER_PATH = CACHE_FOLDER + TRAINING_DATASET_FOLDER + '/'
SYNTHETIC_FOLDER = CACHE_FOLDER + SYNTHETIC_DATASET_FOLDER + '/'
IMAGES_FOLDER = CACHE_FOLDER + IMAGES_FOLDER_NAME
MODEL_RESULTS_FOLDER = RESULTS_FOLDER + IMAGES_FOLDER_NAME
MODEL_OUTPUT_FOLDER = OUTPUT_FOLDER + IMAGES_FOLDER_NAME

# Dataset and model settings
TILE_WIDTH = 200
TILE_HEIGHT = 200
MODEL_FILE = MODELS_FOLDER + MODEL_FILE_NAME
RESULTS_FILE = MODEL_RESULTS_FOLDER + '.csv'
LABELS = ['other', 'transients']

# Training dataset settings
EPOCH = 100
TRAINING_FOLDER = TRAINING_FOLDER_PATH + 'training'
VALIDATION_FOLDER = TRAINING_FOLDER_PATH + 'validation'

# Synthetic dataset settings
DATASET_SOURCE_FOLDER = SYNTHETIC_FOLDER + 'sample' # location where sample dataset is added
DATASET_TEMP_FOLDER = SYNTHETIC_FOLDER + 'temp' 
DATASET_TRAINING_FOLDER = SYNTHETIC_FOLDER + 'training' # location where training images are kept
DATASET_VALIDATION_FOLDER = SYNTHETIC_FOLDER + 'validation'
