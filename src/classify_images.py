"""
Classify images to detect transient objects using a convolutional
neural network model

"""
from __future__ import print_function

import os
import glob
import math
import time
import tflearn
import numpy as np
import pandas as pd
import settings as s
import tensorflow as tf
from scipy import misc

from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

EXPERIMENT_NAME = 'exp6'
MODEL_FILE = s.MODELS_DIRECTORY + 'transients'
IMAGES_FOLDER = s.CACHE_DIRECTORY + 'astrosmall01'
OUTPUT_FOLDER = s.OUTPUT_DIRECTORY + 'test'
CONFIDENCE_THRESHOLD = 0.9
TILE_BRIGHTNESS_THRESHOLD = 180
RESULTS_FILE = s.RESULTS_DIRECTORY + EXPERIMENT_NAME + '.csv'


def main():

    start_time = time.time()
    
    # 
    # 1. Load the dataset
    # 
    data_load_time = time.time()
    print('\nLoad data:')

    # Load annotations
    # Retrieve file locations of images in the dataset
    images = glob.glob(IMAGES_FOLDER + '/*.jpg')

    # Retrieve the class labels
    labels = S.LABELS

    # List to store classification predictions and scores for images
    output_df = pd.DataFrame()

    print('  # images: %d ' % (len(images)))
    print('  # labels: %d ' % (len(labels)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Load pre-trained model
    # 
    model_load_time = time.time()
    print('\nLoad pre-trained model:')

    # Normalise the image data
    image_prep = ImagePreprocessing()
    image_prep.add_featurewise_zero_center()
    image_prep.add_featurewise_stdnorm()

    # Convolutional network
    network = input_data(shape=[None, 200, 200, 1], name='input', data_preprocessing=image_prep)
    network = conv_2d(network, 12, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 24, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 36, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 10, activation='tanh')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    # load the convolutional neural network model
    model = tflearn.DNN(network)
    model.load(MODEL_FILE)

    print('  file: %s' % (MODEL_FILE))
    print('  time taken: %.3f seconds' % (time.time() - model_load_time))


    # 
    # 3. Classify images
    # 
    classify_time = time.time()
    print('\nClassify images:')

    output_df = predict(IMAGES_FOLDER, OUTPUT_FOLDER, CONFIDENCE_THRESHOLD, model)

    # print('  predictions:')
    # for index, value in class_counts_df.iteritems():
    #     print ('    %s: %d' % (index, value))
    print('  time taken: %.3f seconds' % (time.time() - classify_time))


    # 
    # 4. Save prediction results
    # 

    # Reorder the DataFrame columns
    output_df = output_df[['image', 'confidence', 'x0', 'y0', 'x1', 'y1', 'tile']]

    # Write output to results file
    output_df.to_csv(RESULTS_FILE, index=False)


    # 
    # 5. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    print('Generated file:')
    print('  %s\n' % (RESULTS_FILE))


def tile(filename, width, height):
    ''' Create tiles of the image to be used for classification '''
    # Initialise variables
    width_x = 0
    width_y = 0
    image_tiles = []
    image_coordinates = []

    # Load the image    
    image = misc.imread(filename)
    Y = image.shape[0]
    X = image.shape[1]

    # Resize the image if needed
    if ((Y > 1228) or (X > 1840)):
        image = misc.imresize(image, [1228, 1840])
    
    # Number of tiles
    n_rows = int(math.ceil(Y / float(height)))
    n_cols = int(math.ceil(X / float(width)))

    # Divide the image vertically into tiles
    for j in range(n_rows):

        # Make the last tile overlap with the second last tile to ensure its dimensions are 200 x 200 pixels
        if (j == n_rows - 1):
            width_y = (Y - height)
        
        width_x = 0

        # Divide the image horizontally into tiles
        for i in range(n_cols):
            # Make the last tile overlap with the second last tile to ensure its dimensions are 200 x 200 pixels
            if (i == n_cols - 1):
                width_x = (X - width)

            # Conver the image data to float in order to apply normalisation for classification
            image = image.astype('float32')
            image_tiles.append(image[width_y:width_y + height, width_x:width_x + width])
            image_coordinates.append((width_x, width_y, width_x + width, width_y + height))

            width_x += width
        width_y += height

    return image_tiles, image_coordinates


def predict(images_folder, output_folder, threshold, model):
    ''' Returns 1 or 0 based on if transient object is present. If 1, out_filename_tilename is created '''
    # Store classification predictions and scores for images
    results = pd.DataFrame()

    for file in os.listdir(images_folder):
        tiles, coords = tile(images_folder + '/' + file, 200, 200)
        flag = 0

        for i in range(len(tiles)): 
            # Run the model to determine if this tile contains a transient object

            # Calculate the average brightness of the background image tile
            tile_brightness = int(np.mean(tiles[i]))
            
            # Don't classify tiles that are too bright
            if tile_brightness < TILE_BRIGHTNESS_THRESHOLD:                
                score = model.predict([np.reshape(tiles[i], [200, 200, 1])])[0][1]

                if (score >= threshold):
                    tile_row = int(i / 10)
                    tile_column = int(i % 10)

                    filename = output_folder + '/out_' + file + '_' + str(tile_row) + '_' + str(tile_column) + '.jpg'

                    # Append the prediction to the output DataFrame
                    results = results.append({
                        'image': file, 
                        'confidence': round(score, 5),
                        'x0': coords[i][0],
                        'y0': coords[i][1],
                        'x1': coords[i][2],
                        'y1': coords[i][3],
                        'tile': i},
                        ignore_index=True)
                    misc.imsave(filename, tiles[i])
                    flag = 1

        if flag:
            print (file + '--->1')
        else:
            print (file + '--->0')

    return results


if __name__ == '__main__':
    main()
