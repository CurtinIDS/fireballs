"""
Split the dataset images into tiles for fireballs detection

"""
from __future__ import print_function

import time
import os
import fnmatch
import numpy as np
import pandas as pd
import settings as s

from skimage import io


def main():  
    # Initialisation
    start_time = time.time()


    # 
    # 1. Load the dataset
    # 

    data_load_time = time.time()
    print('\nLoad data:')

    # load the data file
    meteors = pd.read_csv(s.DATA_FILE, index_col=False)

    # Retrieve image filenames in the dataset
    images = [[os.path.join(dirpath, f)]
              for dirpath, dirnames, files in os.walk(s.DATA_DIRECTORY)
              for f in fnmatch.filter(files, '*.jpg')] 

    # Create a dataframe to process the images
    df = pd.DataFrame.from_records(images)
    df.rename(columns={0: 'file'}, inplace=True)

    # Extract the relevant information from the image filenames
    df['temp'] = df['file'].str.split('/')
    df['image'] = df['temp'].str[4]
    df['camera'] = df['temp'].str[2]
    df['label'] = df['temp'].str[3]
    df.drop('temp', axis=1, inplace=True)

    print('  # images: %d ' % (len(images)))
    print('  # meteors: %d ' % (len(meteors)))
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
    
    # Generate image tiles
    for index, image in df.iterrows():

        # Load the image
        image_data = io.imread(image['file'])
        
        # Filename info used saving tile images
        filename = image['image']
        filename_parts = filename.split('.')
        
        # Get the bounding box coordinates of meteor(s) in this image
        image_meteors = meteors[meteors['image'] == filename]

        # Generate tiles for each row and column 
        for row in range(rows):
            for col in range(cols):

                # Pixel coordinates for the tile
                x0 = col * width
                x1 = x0 + width
                y0 = row * height
                y1 = y0 + height

                # Default to no meteors for the tile
                label = 'none'

                # Determine if this tile contains a meteor
                if len(image_meteors):
                   
                    # Loop through the list of meteors
                    for index, image_meteor in image_meteors.iterrows():
                        
                        # Compare tile and meteor bounding box coordinates
                        has_meteor = False

                        # Top left grid
                        if (x0 <= image_meteor['x0'] and y0 <= image_meteor['y0'] and 
                                x1 >= image_meteor['x0'] and y1 >= image_meteor['y0']):
                            has_meteor = True
                        # Top grid
                        elif (x0 >= image_meteor['x0'] and y0 <= image_meteor['y0'] and 
                                x1 <= image_meteor['x1'] and y1 >= image_meteor['y0']):
                            has_meteor = True
                        # Top right grid
                        elif (x0 <= image_meteor['x1'] and y0 <= image_meteor['y0'] and 
                                x1 >= image_meteor['x1'] and y1 >= image_meteor['y0']):
                            has_meteor = True
                        # Middle left grid
                        elif (x0 <= image_meteor['x0'] and y0 >= image_meteor['y0'] and 
                                x1 >= image_meteor['x0'] and y1 <= image_meteor['y1']):
                            has_meteor = True    
                        # Middle grid
                        elif (x0 >= image_meteor['x0'] and y0 >= image_meteor['y0'] and 
                                x1 <= image_meteor['x1'] and y1 <= image_meteor['y1']):
                            has_meteor = True
                        # Middle right grid
                        elif (x0 <= image_meteor['x1'] and y0 >= image_meteor['y0'] and 
                                x1 >= image_meteor['x1'] and y1 <= image_meteor['y1']):
                            has_meteor = True    
                        # Bottom left grid
                        elif (x0 <= image_meteor['x0'] and y0 <= image_meteor['y1'] and 
                                x1 >= image_meteor['x0'] and y1 >= image_meteor['y1']):
                            has_meteor = True
                        # Bottom grid
                        elif (x0 >= image_meteor['x0'] and y0 <= image_meteor['y1'] and 
                                x1 <= image_meteor['x1'] and y1 >= image_meteor['y1']):
                            has_meteor = True
                        # Bottom right grid
                        elif (x0 <= image_meteor['x1'] and y0 <= image_meteor['y1'] and
                                x1 >= image_meteor['x1'] and y1 >= image_meteor['y1']):
                            has_meteor = True

                        if has_meteor:
                            label = 'meteors'

                # Generate the tile filename 
                tile_filename = s.CACHE_DIRECTORY + image['camera'] + '/' + label + '/'
                tile_filename += filename_parts[0] + '_' + str(row) + str(col)
                tile_filename += '.' + filename_parts[1] + '.' + filename_parts[2]

                # Save the tile image to disk
                io.imsave(tile_filename, image_data[y0:y1, x0:x1, :])

    print('  tiles created: %d images' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - image_tiling_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
