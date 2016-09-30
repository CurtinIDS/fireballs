"""
Move images containing transisent objects to their corresponding labelled folder

"""
from __future__ import print_function

import time
import os
import fnmatch
import numpy as np
import pandas as pd
import settings as s


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

    # Remove duplicate records of the same image. Images can be listed multiple 
    # times as a single image can contain multiple transisent objects
    df.drop_duplicates(subset={'image'}, keep='first', inplace=True)

    print('  # records: %d ' % (initial_df_len))
    print('  # unique transient object images: %d ' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Move transisent object image files
    # 
    data_prep_time = time.time()
    print('\nPrepare data:')

    # Move each file
    moved_images = 0
    for index, transient in df.iterrows():
        # all images are initially stored located in the other folder
        existing_file = s.DATA_DIRECTORY + transient['camera'] + '/other/' + transient['image']
        new_file = s.DATA_DIRECTORY + transient['camera'] + '/transients/' + transient['image']

        try:
            os.rename(existing_file, new_file)
            moved_images += 1
        except OSError:
            print('unable to move: ', existing_file, new_file)
            pass


    print('  move labelled files to transients folder: %d images' % (moved_images))
    print('  time taken: %.3f seconds' % (time.time() - data_prep_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
