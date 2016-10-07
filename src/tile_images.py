"""
Split the dataset images into tiles for fireballs detection

This script is used for when tile coordinates have been precomputed and populated 
to data/transients.csv. Otherwise, run the initial_tile_images.py and meteor_grids.py 
scripts to extract tile coordinates for images labelled as containing meteor(s).

"""
from __future__ import print_function

import time
import os
import fnmatch
import pandas as pd
import settings as s
import multiprocessing
import joblib

from skimage import io


def main():  
    # Initialisation
    start_time = time.time()


    # 
    # 1. Load the dataset
    # 

    data_load_time = time.time()
    print('\nLoad data:')

    # load the transients data file
    transients = pd.read_csv(s.DATA_FILE, index_col=False)

    # Retrieve image filenames in the dataset directory
    images = [[os.path.join(dirpath, f)]
              for dirpath, dirnames, files in os.walk(s.DATA_DIRECTORY)
              for f in fnmatch.filter(files, '*.jpg')] 

    # Create a dataframe to process the images
    df = pd.DataFrame.from_records(images)
    df.rename(columns={0: 'file'}, inplace=True)

    # Extract relevant information from the image filenames
    df['temp'] = df['file'].str.split('/')
    df['image'] = df['temp'].str[4]
    df['camera'] = df['temp'].str[2]
    df['label'] = df['temp'].str[3]
    df.drop('temp', axis=1, inplace=True)

    # Filter tiling to a specific camera
    # df = df[df['camera'] == s.CAMERAS[0]]
    
    # Only select a number of non meteor images for creating background tiles
    required_images = 50

    # Created a background tiles dataframe containing only selected images
    background_df = df[df['label'] == s.LABEL_OTHER]
    step = len(background_df) / required_images
    indices = [i * step for i in range(required_images)]
    background_df = background_df.iloc[indices]

    # Append the selected background images to the transients dataframe
    df = df[df['label'] == s.LABEL_TRANSIENT]
    df = df.append(background_df)
  
    print('  # images: %d ' % (len(images)))
    print('  # transient objects: %d ' % (len(transients)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Generate image tiles
    # 

    image_tiling_time = time.time()
    print('\nGenerate image tiles:')

    # Get folder names
    cameras = df['camera'].unique()
    labels = df['label'].unique()

    # Create the folders in the cache directory if they don't already exist
    for camera in cameras:
        for label in labels:
            cache_folder = s.CACHE_DIRECTORY + camera + '/' + label

            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder) 
                pass

    # Generate 10 rows x 10 columns of tiles for images
    rows = 10
    cols = 10

    # Get the dimensions of each tile based on the first image
    image = io.imread(df.iloc[0]['file'])
    height = int(image.shape[0] / rows)
    width = int(image.shape[1] / cols)
    
    # 
    # Parallelised implementation
    # 
    # Benchmarks
    # Parallel = 8 cores threading
    # - 10 images
    #   - Serial: 72 secs (1m 12s)
    #   - Parallel: 28 secs
    # - 50 images
    #   - Serial:  359 secs (5m 59s)
    #   - Parallel: 124 secs (2m 4s)
    # - 100 images
    #   - Serial: 728 secs (12m 8s)
    #   - Parallel: 240 secs (4m 0s)
    # parallel_time = time.time() 

    # Generate image tiles
    generate_tiles(df, rows, cols, height, width, transients)

    # print('  parallel: %.3f seconds' % (time.time() - parallel_time))

    # 
    # Serial implementation
    # 
    # serial_time = time.time()

    # for index, image in df.iterrows():

    #     # Load the image
    #     image_data = io.imread(image['file'])
        
    #     # Filename info used saving tile images
    #     filename = image['image']
    #     filename_parts = filename.split('.')
        
    #     # Check if this image contains transients
    #     image_transients = transients[transients['image'] == filename]

    #     # Tile coordinates are known from the transients data file
    #     meteor_tiles = ','.join(image_transients['tiles'].values).strip().split(',')
        
    #     # Generate tiles for each row and column 
    #     for row in range(rows):
    #         for col in range(cols):

    #             # Pixel coordinates for the tile
    #             x0 = col * width
    #             x1 = x0 + width
    #             y0 = row * height
    #             y1 = y0 + height

    #             # Default to no transients category for the tile
    #             label = s.LABEL_OTHER

    #             # Image has been labelled to contain meteor(s)
    #             if len(image_transients):

    #                 # Check if this tile has been labelled as a meteor tile
    #                 for tile in meteor_tiles:
    #                     if row == int(tile[0]) and col == int(tile[1]):
    #                         # Remove the tile once it has been found
    #                         meteor_tiles.remove(tile)
    #                         # Set the tile label as a meteor tile
    #                         label = 'transients'

    #             # Generate the tile filename 
    #             tile_filename = s.CACHE_DIRECTORY + image['camera'] + '/' + label + '/'
    #             tile_filename += filename_parts[0] + '_' + str(row) + str(col)
    #             tile_filename += '.' + filename_parts[1] + '.' + filename_parts[2]

    #             # Save the tile image to the correct labelled folder
    #             io.imsave(tile_filename, image_data[y0:y1, x0:x1, :])

    # print('  serial: %.3f seconds' % (time.time() - serial_time))

    print('  tiles created: %d images' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - image_tiling_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


def generate_tiles(images, rows, cols, height, width, transients):
    ''' Outer loop for tiling images '''
    # Use the maximum number of cores available
    num_cores = multiprocessing.cpu_count()
    par = joblib.Parallel(n_jobs=num_cores, backend="threading")

    # function to be parallelised
    par_func = joblib.delayed(_generate_tiles_inner)

    # run the parallelised function
    par(par_func(image, rows, cols, height, width, transients) for index, image in images.iterrows())
    

def _generate_tiles_inner(image, rows, cols, height, width, transients):
    ''' Inner loop for tiling an image '''
    # Load the image
    image_data = io.imread(image['file'])
    
    # Extract filename info used for saving tile images
    filename = image['image']
    filename_parts = filename.split('.')
    
    # Check if this image contains transients
    image_transients = transients[transients['image'] == filename]

    # Tile coordinates are known from the transients data file
    meteor_tiles = ','.join(image_transients['tiles'].values).strip().split(',')
    
    # Generate tiles for each row and column 
    for row in range(rows):
        for col in range(cols):

            # Pixel coordinates for the tile
            x0 = col * width
            x1 = x0 + width
            y0 = row * height
            y1 = y0 + height

            # Default to no transients category for the tile
            label = s.LABEL_OTHER

            # Image has been labelled to contain meteor(s)
            if len(image_transients):

                # Check if this tile has been labelled as a meteor tile
                for tile in meteor_tiles:
                    if row == int(tile[0]) and col == int(tile[1]):
                        # Remove the tile once it has been found
                        meteor_tiles.remove(tile)
                        # Set the tile label as a meteor tile
                        label = 'transients'

            # Generate the tile filename 
            tile_filename = s.CACHE_DIRECTORY + image['camera'] + '/' + label + '/'
            tile_filename += filename_parts[0] + '_' + str(row) + str(col)
            tile_filename += '.' + filename_parts[1] + '.' + filename_parts[2]

            # Save the tile image to the correct labelled folder
            io.imsave(tile_filename, image_data[y0:y1, x0:x1, :])


if __name__ == '__main__':
    main()
