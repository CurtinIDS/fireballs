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
IMAGES_FOLDER = s.CACHE_FOLDER + IMAGES_FOLDER_NAME
RESULTS_FOLDER = s.OUTPUT_FOLDER + s.IMAGES_FOLDER_NAME
OUTPUT_FOLDER = s.OUTPUT_FOLDER + s.IMAGES_FOLDER_NAME

# Dataset and model settings
TILE_WIDTH = 200
TILE_HEIGHT = 200
MODEL_FILE = s.MODELS_DIRECTORY + MODEL_FILE_NAME
RESULTS_FILE = s.RESULTS_FOLDER + s.IMAGES_FOLDER_NAME + '.csv'
LABELS = ['other', 'transients']

# Synthetic dataset settings
SOURCE_FOLDER = s.SYNTHETIC_FOLDER + 'source'
TEMP_FOLDER = s.SYNTHETIC_FOLDER + 'temp' 
TRAINING_FOLDER = s.SYNTHETIC_FOLDER + 'training'
VALIDATION_FOLDER = s.SYNTHETIC_FOLDER + 'validation'
