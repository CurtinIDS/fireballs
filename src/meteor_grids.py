"""
Script used for extracting the identified tile coordinates 
of meteor images based on their filenames
"""
from __future__ import print_function

import os
import fnmatch
import time
import settings as s


def main():

    start_time = time.time()
    
    # 
    # 1. Load the dataset
    # 
    data_load_time = time.time()
    print('\nLoad data:')

    meteors_directory = s.IMAGES_DIRECTORY + 'meteors/' 

    print (meteors_directory)

    # Retrieve image filenames in the dataset
    images = [os.path.join(dirpath, f)
              for dirpath, dirnames, files in os.walk(meteors_directory)
              for f in fnmatch.filter(files, '*.jpg')]

    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Extract meteor tile / grid coordinates 
    # 

    # Initialise grid coordinates list
    grids = []
    # Set the current image to the first image
    parent_image = images[0].split('/')[4].split('.')[0][:-3]

    # Examine each meteor image tile 
    # Note: a meteor can span over multiple tiles but these tiles are taken
    # from the same parent image
    for image in images:

        # Extract information from the filename
        image_file = image.split('/')[4].split('.')[0]                
        current_parent_image = image_file[:-3]
        grid = image_file[-2:]
        
        # Image tile belongs to same parent image as the previous evaluated tile
        if parent_image == current_parent_image:
            # Append the grid coordinates
            grids.append(grid)
        # New parent image
        else: 
            # Print all grid coordinates for the last parent image
            print(','.join(grids), '\t', parent_image)
            # Re-initialise variables for new parent image
            grids = []
            parent_image = current_parent_image
            # Append the grid coordinates
            grids.append(grid)

    
    # Print the coordinates for the last image
    print(','.join(grids), '\t', parent_image)

    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    print('Generated file:')
    print('  %s\n' % (s.RESULTS_FILE))


if __name__ == '__main__':
    main()
