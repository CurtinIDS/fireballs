"""
Global settings for the project

"""
# Directory paths are relative to the src folder
DATA_DIRECTORY = '../data/'
CACHE_DIRECTORY = '../cache/'
OUTPUT_DIRECTORY = '../output/'
RESULTS_DIRECTORY = '../results/'
# Experiment
CAMERAS = ['astrosmall00_mobile', 'astrosmall01', 'dfnsmall33', 'dfnsmall49']
EXPERIMENT_NAME = CAMERAS[0]
# EXPERIMENT_NAME = 'all'
EXPERIMENT_TYPE = 'retrained'
EXPERIMENT_PREFIX = EXPERIMENT_NAME + '_' + EXPERIMENT_TYPE
TRAINING_STEPS = 10000
BOTTLENECK_DIRECTORY = OUTPUT_DIRECTORY + 'bottlenecks' 
INCEPTION_DIRECTORY = OUTPUT_DIRECTORY + 'inception'
MODEL_GRAPH_FILE = OUTPUT_DIRECTORY + EXPERIMENT_PREFIX + '_graph.pb'
IMAGE_LABELS_FILE = OUTPUT_DIRECTORY + EXPERIMENT_PREFIX + '_labels.txt'
# Data
# DATA_FILE = DATA_DIRECTORY + 'meteors.csv'
DATA_FILE = DATA_DIRECTORY + 'transients.csv'
IMAGES_DIRECTORY = CACHE_DIRECTORY + EXPERIMENT_NAME + '/'
RESULTS_FILE = RESULTS_DIRECTORY + EXPERIMENT_PREFIX + '_experiment_results.csv'
RESULTS_SUMMARY_FILE = RESULTS_DIRECTORY + EXPERIMENT_PREFIX + '_experiment_summary.txt'
