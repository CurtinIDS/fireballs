"""
Global settings for the project

"""
# User defined settings
IMAGES_FOLDER_NAME = 'test'
MODEL_FILE_NAME = 'transients'
TRAINING_DATASET_FOLDER = 'temp'
SYNTHETIC_DATASET_FOLDER = 'synthetic'
# Set the training dataset folder to the synthetic dataset folder if
# you want to train on the generated dataset
# TRAINING_DATASET_FOLDER = SYNTHETIC_DATASET_FOLDER

# Directory paths are relative to the src folder
DATA_FOLDER = '../data/'
CACHE_FOLDER = '../cache/'
OUTPUT_FOLDER = '../output/'
RESULTS_FOLDER = '../results/'
MODELS_DIRECTORY = '../models/'
SYNTHETIC_FOLDER = CACHE_FOLDER + SYNTHETIC_DATASET_FOLDER
IMAGES_FOLDER = CACHE_FOLDER + IMAGES_FOLDER_NAME
MODEL_RESULTS_FOLDER = RESULTS_FOLDER + IMAGES_FOLDER_NAME
MODEL_OUTPUT_FOLDER = OUTPUT_FOLDER + IMAGES_FOLDER_NAME

# Dataset and model settings
TILE_WIDTH = 200
TILE_HEIGHT = 200
MODEL_FILE = MODELS_DIRECTORY + MODEL_FILE_NAME
RESULTS_FILE = MODEL_RESULTS_FOLDER + '.csv'
LABELS = ['other', 'transients']

# Training dataset settings
TRAINING_FOLDER = TRAINING_DATASET_FOLDER + 'training'
VALIDATION_FOLDER = TRAINING_DATASET_FOLDER + 'validation'

# Synthetic dataset settings
DATASET_SOURCE_FOLDER = SYNTHETIC_FOLDER + 'source'
DATASET_TEMP_FOLDER = SYNTHETIC_FOLDER + 'temp' 
DATASET_TRAINING_FOLDER = SYNTHETIC_FOLDER + 'training'
DATASET_VALIDATION_FOLDER = SYNTHETIC_FOLDER + 'validation'
