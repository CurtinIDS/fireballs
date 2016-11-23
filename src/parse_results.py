"""
Parse the fireballs detection results and highlight tiles containing 
transient objects within the images. 

"""
from __future__ import print_function

import os
import shutil
import time
import pandas as pd
import numpy as np
import settings as s

from scipy import misc

FOLDER_NAME = 'astrosmall01'
SOURCE_FOLDER = s.CACHE_DIRECTORY + FOLDER_NAME + '/'
RESULTS_FOLDER = s.RESULTS_DIRECTORY + FOLDER_NAME
RESULTS_FILE = s.RESULTS_DIRECTORY + FOLDER_NAME + '.csv'
# Brightness factor to increase tiles containing transient objects 
BRIGHTNESS_FACTOR = 30


def main(): 
    start_time = time.time()

    # 
    # 1. Load and prepare the dataset 
    # 
    data_load_time = time.time()
    print('\nLoad results data and prepare images:')

    # Read the results file
    df = pd.read_csv(RESULTS_FILE)
 
    # Create a folder to store the results annotated images
    create_folder(RESULTS_FOLDER)

    # Copy the existing files across
    images = df['image'].unique()
    copy_files(images, SOURCE_FOLDER, RESULTS_FOLDER)

    print('  file: %s' % (RESULTS_FILE))
    print('  # images: %d' % (len(images)))
    print('  # tiles: %d' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Annotate the images
    # 
    annotate_time = time.time()
    print('\nAnnotate images:')

    # Get the dimensions of each tile based on the first image
    image = misc.imread(RESULTS_FOLDER + '/' + df.iloc[0]['image'])
    height = int(image.shape[0])
    width = int(image.shape[1])

    # Determine the amount of overlap of tiles at the bottom or the right side of the image
    overlap_height = height % s.TILE_HEIGHT
    overlap_width = width % s.TILE_WIDTH

    # Increase the brightnness of the selected coordinates /tiles
    for image in images:
    
        # Load the image
        image_filename = RESULTS_FOLDER + '/' + image
        image_data = misc.imread(image_filename)

        # retrieve all the tiles containing transient objects in the image
        tiles = df[df['image'] == image]

        # Increase the brightness of each tile
        for row, tile in tiles.iterrows():          
            # Extract the coordinates
            x0 = int(tile['x0'])
            y0 = int(tile['y0'])
            x1 = int(tile['x1'])
            y1 = int(tile['y1'])

            # Check if we are evaluating tiles at the bottom or right side of the image
            if height == y1 or width == x1:
                # Check if the above tile is already highlighted
                above_tile_y1 = int(height - overlap_height)
                above_tile = tiles[(tiles['x0'] == x0) & (tiles['y1'] == above_tile_y1) & (tiles['tile'] != tile['tile'])]

                # Check if the left tile is already highlighted
                left_tile_x1 = int(width - overlap_width)
                left_tile = tiles[(tiles['y0'] == y0) & (tiles['x1'] == left_tile_x1) & (tiles['tile'] != tile['tile'])]
            
                # Adjust the highlighting height for the overlapping tile    
                if len(above_tile):
                    y0 = above_tile_y1
                
                # Adjust the highlighting width for the overlap tile
                if len(left_tile):
                    x0 = left_tile_x1

            # Increase the brightness of the tile
            image_data[y0:y1, x0:x1] += BRIGHTNESS_FACTOR

        misc.imsave(image_filename, image_data)

    print('  tiles annotated: %d seconds' % (len(df)))   
    print('  time taken: %.3f seconds' % (time.time() - annotate_time))


    # 
    # 3. Display messages to the console
    #
 
    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))
    
    print('Annotated images folder:')
    print('  %s/\n' % (RESULTS_FOLDER))


def create_folder(name):
    ''' Create the folder for the annotated images '''
    if os.path.exists(name):
        # Remove the existing folder
        shutil.rmtree(name)
    
    # Create the folder
    os.makedirs(name)


def copy_files(files, source, destination):
    ''' Copy the source images to the annotated results folder '''  
    for file in files:
        shutil.copy(os.path.join(source, file), destination)


if __name__ == '__main__':
    main()
