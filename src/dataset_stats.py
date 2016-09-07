"""
Calculate statistics on the fireballs dataset

"""
from __future__ import print_function

import time
import pandas as pd
import settings as s
import os
import fnmatch


def main():  
    # Initialisation
    start_time = time.time()


    # 
    # 1. Load and prepare the dataset
    # 

    data_load_time = time.time()
    print('\nLoad data:')

    # Retrieve image filenames in the dataset
    images = [[os.path.join(dirpath, f)]
              for dirpath, dirnames, files in os.walk(s.DATA_DIRECTORY)
              for f in fnmatch.filter(files, '*.jpg')] 

    # Create a dataframe to process the images
    df = pd.DataFrame.from_records(images)
    df.rename(columns={0: 'file'}, inplace=True)

    # Extract the relevant information from the image filenames
    df['temp'] = df['file'].str.split('/')
    df['camera'] = df['temp'].str[2]
    df['label'] = df['temp'].str[3]
    df.drop('temp', axis=1, inplace=True)

    print('  # records: %d ' % (len(images)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Calculate dataset statistics on the fireballs classification labels
    #     
    
    stats_time = time.time()
    print('\nCalculate dataset classification stats:')

    # Count the number images for each camera and label
    camera_counts_df = df.groupby('camera')['camera'].size()
    label_counts_df = df.groupby(['camera', 'label'])['camera'].size()

    # Display the dataset stats information
    camera = ''
    for index, value in label_counts_df.iteritems():
        # Only display the camera counts once per camera
        if camera != index[0]:
            camera = index[0]
            print ('  %s: %d' % (camera, camera_counts_df.loc[camera]))

        # Display the label counts for each camera
        print ('    %s: %d' % (index[1], value))

    print('  time taken: %.3f seconds' % (time.time() - stats_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
