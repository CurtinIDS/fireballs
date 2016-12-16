"""
Create a dataset that consists of transient objects (streaks) synthetically generated
and placed on real background image tiles of the night sky

"""
import math
import numpy as np
import os
import shutil
import settings as s
import time
from random import random, seed, randrange
from shutil import copyfile
from PIL import Image, ImageDraw

# Number of samples to generate
TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = int(TRAINING_SAMPLES * 0.1)
# Transient / no transient ratio
BIAS = 0.5
# Brightness of the transient objects (streaks) drawn
BRIGHTNESS_VALUES = [250, 200, 150, 100, 100]
BRIGHTNESS_THRESHOLD_VALUES = [0.35, 0.40, 0.45, 0.70, 0.70]
STREAK_BRIGHTNESS_INDEX = 0
STREAK_BRIGHTNESS = BRIGHTNESS_VALUES[STREAK_BRIGHTNESS_INDEX]
BRIGHTNESS_THRESHOLD = BRIGHTNESS_THRESHOLD_VALUES[STREAK_BRIGHTNESS_INDEX]
# Seed numbers used for random number generator to repliciating experimental results
TRAINING_SEED = STREAK_BRIGHTNESS_INDEX + 5656
VALIDATION_SEED = STREAK_BRIGHTNESS_INDEX + 2961


def main(): 
    # Initialisation
    start_time = time.time()


    # 
    # 1. Prepare the dataset 
    # 
    data_prep_time = time.time()
    print('\nPrepare dataset:')

    # Create folders for generating storing generated synthetic images 
    create_folder(s.DATASET_TEMP_FOLDER)
    create_folder(s.DATASET_TRAINING_FOLDER, labels=True)
    create_folder(s.DATASET_VALIDATION_FOLDER, labels=True)

    # Copy source images to a temporary folder
    count = 0
    for image in os.listdir(s.DATASET_SOURCE_FOLDER):
        copyfile(s.DATASET_SOURCE_FOLDER + '/' + image, s.DATASET_TEMP_FOLDER + '/' + str(count) + '.jpg')
        count += 1

    print('  # source images: %d ' % (count - 1))
    print('  time taken: %.3f seconds' % (time.time() - data_prep_time))


    # 
    # 2. Generate training dataset images
    # 
    training_start_time = time.time()
    print('\nGenerate training images:')

    generate_images(s.DATASET_TRAINING_FOLDER, TRAINING_SAMPLES, TRAINING_SEED)

    print('  # training images generated: %d ' % (TRAINING_SAMPLES))
    print('  time taken: %.3f seconds' % (time.time() - training_start_time))


    # 
    # 3. Generate validation dataset images
    #     
    validation_start_time = time.time()
    print('\nGenerate validation images:')

    generate_images(s.DATASET_VALIDATION_FOLDER, VALIDATION_SAMPLES, VALIDATION_SEED)

    print('  # validation images generated: %d ' % (VALIDATION_SAMPLES))
    print('  time taken: %.3f seconds' % (time.time() - validation_start_time))


    # 
    # 4. Display messages to the console
    #
    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    
def generate_images(folder, samples, random_seed):
    ''' Generate synthethic images '''

    # Seed the random number generator   
    seed(random_seed)
    image_len = len(os.listdir(s.DATASET_SOURCE_FOLDER))
 
    for i in range(samples):

        # Select a random background image without meteorites
        myrand = randrange(1, image_len)

        # 0.008 - 0.013 seconds to load file
        im = Image.open(s.DATASET_TEMP_FOLDER + '/' + str(myrand) + '.jpg')
        X = im.size[0]
        Y = im.size[1]
        
        # Resize the image if required 
        if ((X > 1840) or (Y > 1228)):
            im = im.thumbnail(1840, 1228)

        width, height = im.size
        marker_x = width * 2
        marker_y = height * 2

        # Randomly select the area within the image that will be used as the background tile
        while ((marker_x + s.TILE_WIDTH) > width):
            marker_x = int(random() * width)
            while ((marker_y + s.TILE_HEIGHT) > height):
                marker_y = int(random() * height)

        # 0.034 - 0.039 seconds (slower operation on Mac OSX than Linux)
        im = im.crop((marker_x, marker_y, marker_x + s.TILE_WIDTH, marker_y + s.TILE_HEIGHT))
        draw = ImageDraw.Draw(im)
        A = [(random() * im.size[0], random() * im.size[1], random() * im.size[0], random() * im.size[1])]
        myrand = str(myrand)

        # Calculate the average brightness of the background image tile
        tile_brightness = int(np.mean(im))

        has_meteor = False

        if (random() > BIAS):
            # The lines shouldn't be too short or too long
            if ((dist(A) > 30) and (dist(A) < 300)): 

                color_rnd = 0
                # Brightness should be over certain threshold, lower the brightness -> harder to train, more resilient
                while (color_rnd < BRIGHTNESS_THRESHOLD): 
                    color_rnd = random() 

                streak_brightness = color_rnd * STREAK_BRIGHTNESS
                # Ensure the streak is brighter than the tile average brightness
                if tile_brightness > streak_brightness:
                    streak_brightness = min(tile_brightness + 40, 255)
                # Enhance the streak brightness if the difference between the tile average is small
                elif streak_brightness - tile_brightness < 35:
                    streak_brightness = min(streak_brightness + 35, 255)
                
                width_rand = 0
                while (width_rand < 0.5):
                    # Width should be less than 2 px
                    width_rand = int(random() * 2) 

                draw.line([A[0][0], A[0][1], A[0][2], A[0][3]], fill=int(streak_brightness), width=width_rand)
                # Add _y to files that contain transient objects
                myrand = myrand + '_y' 
                has_meteor = True
                del draw
                

        if has_meteor:
            im.save(folder + '/1/' + str(i) + '_' + str(tile_brightness) + '_' + str(myrand) + '.jpg')
        else:
            im.save(folder + '/0/' + str(i) + '_' + str(tile_brightness) + '_' + str(myrand) + '.jpg')


def dist(A): 
    ''' Computes distance between two points in pixels '''
    return(((A[0][0] - A[0][2])**2 + (A[0][1] - A[0][3])**2)**0.5)


def create_folder(name, labels=False):
    ''' Create the folder for generating synthetic images '''
    if os.path.exists(name):
        # Remove the existing folder
        shutil.rmtree(name)
    
    # Create the folder
    os.makedirs(name)
    # Create label folders if required
    if labels:
        os.makedirs(name + '/0')
        os.makedirs(name + '/1')
    

if __name__ == '__main__':
    main()
