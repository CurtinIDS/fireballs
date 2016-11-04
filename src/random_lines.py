#!/usr/bin/env python
# Create new directories

import math
import os
from random import random
from shutil import copyfile
from PIL import Image, ImageDraw


TILE_WIDTH = 200
TILE_HEIGHT = 200
# number of many samples to generate
SAMPLE_SIZE = 1000 
# Transient / no transient ratio
BIAS = 0.5
# original images with _no_ transient objects
SOURCE_FOLDER = 'NONE'
TEMP_FOLDER = 'new' 
OUTPUT_FOLDER = 'post'


def main(): 
    create_dir(TEMP_FOLDER)
    create_dir(OUTPUT_FOLDER)

    count = 0

    for each in os.listdir(SOURCE_FOLDER):
        copyfile(SOURCE_FOLDER + '/' + each, TEMP_FOLDER + '/' + str(count) + '.jpg')
        count += 1
        

    image_len = len(os.listdir(SOURCE_FOLDER))

    for i in range(SAMPLE_SIZE):
        myrand = 0
        while myrand == 0:
            # Open files with random background and w/o meteorites
            myrand = int(random() * image_len)
        
        im = Image.open(TEMP_FOLDER + '/' + str(myrand) + '.jpg')
        X = im.size[0]
        Y = im.size[1]
        if ((X > 1840) or (Y > 1228)):
            im = im.thumbnail(1840, 1228)
        width, height = im.size
        marker_x = width * 2
        marker_y = height * 2
        while ((marker_x + TILE_WIDTH) > width):
            marker_x = int(random() * width)
            while ((marker_y + TILE_HEIGHT) > height):
                marker_y = int(random() * height)
        im = im.crop((marker_x, marker_y, marker_x + TILE_WIDTH, marker_y + TILE_HEIGHT))
        draw = ImageDraw.Draw(im)
        A = [(random() * im.size[0], random() * im.size[1], random() * im.size[0], random() * im.size[1])]
        myrand = str(myrand)
        if (random() > BIAS):
            # The lines shouldn't be too short or too long
            if ((dist(A) > 30) and (dist(A) < 300)): 
                color_rnd = 0
                while (color_rnd < 0.3): 
                    # Brightness should be over certain threshold, lower the brightness -> harder to train, more resilient
                    color_rnd = random() 
                width_rand = 0
                while (width_rand < 0.5):
                    # Width should be less than 2 px
                    width_rand = int(random() * 2) 
                draw.line([A[0][0], A[0][1], A[0][2], A[0][3]], fill=int(color_rnd * 100), width=width_rand)
                # Add _y to files that contain meteors
                myrand = myrand + '_y' 
                del draw

        im.save(OUTPUT_FOLDER + '/' + str(i) + '_' + str(myrand) + '.jpg')


def dist(A): 
    ''' Computes distance between two points in px '''
    return(((A[0][0] - A[0][2])**2 + (A[0][1] - A[0][3])**2)**0.5)


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    

if __name__ == '__main__':
    main()
