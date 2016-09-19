"""
Prepare the dataset for fireballs detection 

"""
from __future__ import print_function

import time
import os
import pandas as pd
import settings as s
import fnmatch


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

    print('  removed from non-meteor folder: %d images' % (removed_images))
    print('  time taken: %.3f seconds' % (time.time() - data_prep_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
