"""
Global settings for the project

"""
# Directory paths are relative to the src folder
DATA_DIRECTORY = '../data/'
CACHE_DIRECTORY = '../cache/'
OUTPUT_DIRECTORY = '../output/'
RESULTS_DIRECTORY = '../results/'
MODELS_DIRECTORY = '../models/'
SYNTHETIC_DIRECTORY = CACHE_DIRECTORY + 'synthetic/'
# Shared dataset and model settings
TILE_WIDTH = 200
TILE_HEIGHT = 200

# Experiment
CAMERAS = ['astrosmall00_mobile', 'astrosmall01']
EXPERIMENT_NAME = CAMERAS[0]
LABELS = ['other', 'transients']
LABEL_OTHER = LABELS[0]
LABEL_TRANSIENT = LABELS[1]
