"""
Prepare the dataset for fireballs detection 

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
    df = pd.read_csv(s.DATA_FILE, index_col=False)
    initial_df_len = len(df)

    print('  # meteor records: %d ' % (initial_df_len))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Data preparation
    # 

    #    
    # 2a. Remove meteor images from the not meteors folder
    # 
    data_prep_time = time.time()
    print('\nPrepare data:')

    # Get the camera folder names
    cameras = df['camera'].unique()

    # Number of files removed after data preparation
    removed_images = 0

    # Data check:
    # Remove meteor images that exist in "none" (not meteors) folders for each camera
    for camera in cameras:
        
        # Not meteors images folder for the camera
        directory_path = s.DATA_DIRECTORY + camera + '/none/'

        # Initial folder file count
        file_count = len(fnmatch.filter(os.listdir(directory_path), '*.jpg'))

        # Modify the image column to include the filepath to the not meteors folder
        camera_images = df[df['camera'] == camera].copy()
        camera_images['image'] = directory_path + camera_images['image'].astype(str)

        # Remove the images
        images = camera_images['image'].tolist()
        for image in images:
            try:
                os.unlink(image)
            except OSError:
                pass

        # Track the number of images removed
        removed_images += (file_count - len(fnmatch.filter(os.listdir(directory_path), '*.jpg')))


    #    
    # 2b. Crop meteor images based on bounding box coordinates
    # 
   
    # Modify the image column to include the filepath to the not meteors folder    
    df['file'] = s.DATA_DIRECTORY + 'raw/' + df['camera'] + '/meteors/' + df['image'].astype(str)

    # Create the folders in the cache directory if they don't already exist
    for camera in cameras:
        camera_folder = s.CACHE_DIRECTORY + camera + '/meteors'

        if not os.path.exists(camera_folder):
            os.makedirs(camera_folder) 

    for index, meteor in df.iterrows():        
        # Read in the meteor image
        image = io.imread(meteor['file'])

        # Crop the image based on the bounding box coordinates
        cropped_image = image[meteor['y1']:meteor['y2'], meteor['x1']:meteor['x2'], :]

        # Output filename
        filename = s.CACHE_DIRECTORY + meteor['camera'] + '/meteors/'
        
        # add a filename suffix for the images containing more than one meteor 
        if np.isnan(meteor['suffix']):
            filename += meteor['image']        
        else:
            filename_parts = meteor['image'].split('.')
            filename += filename_parts[0] + '_' + str(int(meteor['suffix'])) + '.' + filename_parts[1] + '.' + filename_parts[2]
            
        # Save the cropped image to disk
        io.imsave(filename, cropped_image)

    print('  removed from non-meteor folder: %d images' % (removed_images))
    print('  meteors cropped: %d images' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - data_prep_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
