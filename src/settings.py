"""
Global settings for the project

"""
# User defined settings
IMAGES_FOLDER_NAME = 'test'
MODEL_FILE_NAME = 'transients'

# Directory paths are relative to the src folder
DATA_FOLDER = '../data/'
CACHE_FOLDER = '../cache/'
OUTPUT_FOLDER = '../output/'
RESULTS_FOLDER = '../results/'
MODELS_DIRECTORY = '../models/'
SYNTHETIC_FOLDER = CACHE_FOLDER + 'synthetic/'
IMAGES_FOLDER = CACHE_FOLDER + IMAGES_FOLDER_NAME
MODEL_RESULTS_FOLDER = RESULTS_FOLDER + IMAGES_FOLDER_NAME
MODEL_OUTPUT_FOLDER = OUTPUT_FOLDER + IMAGES_FOLDER_NAME

# Dataset and model settings
TILE_WIDTH = 200
TILE_HEIGHT = 200
MODEL_FILE = MODELS_DIRECTORY + MODEL_FILE_NAME
RESULTS_FILE = MODEL_RESULTS_FOLDER + '.csv'
LABELS = ['other', 'transients']

# Synthetic dataset settings
SOURCE_FOLDER = SYNTHETIC_FOLDER + 'source'
TEMP_FOLDER = SYNTHETIC_FOLDER + 'temp' 
TRAINING_FOLDER = SYNTHETIC_FOLDER + 'training'
VALIDATION_FOLDER = SYNTHETIC_FOLDER + 'validation'
