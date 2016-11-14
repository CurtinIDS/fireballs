"""
Classify images to detect transient objects using a convolutional
neural network model

"""
from __future__ import print_function

import os
import fnmatch
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

# Instructions
# -resize to 25% of the original image
# -change predict function towards the end to include test director
#
MODEL_FILE = s.MODELS_DIRECTORY + 'synthetic'
IMAGES_FOLDER = s.CACHE_DIRECTORY + 'astrosmall00_mobile'
OUTPUT_FOLDER = s.OUTPUT_DIRECTORY + 'test'

# if checkpoint file found, then load the model
# model = tflearn.DNN(network, checkpoint_path='.ckpt')
# predict(TEST_FOLDER, model, 0.9)


def main():

    start_time = time.time()
    
    # 
    # 1. Load the dataset
    # 
    data_load_time = time.time()
    print('\nLoad data:')

    # Load annotations
    # Retrieve file locations of images in the dataset
    # images = [os.path.join(dirpath, f) 
    #           for dirpath, dirnames, files in os.walk(s.IMAGES_DIRECTORY)
    #           for f in fnmatch.filter(files, '*.jpg')]

    # Retrieve the class labels
    # labels = [line.rstrip() for line in tf.gfile.GFile(s.IMAGE_LABELS_FILE)]

    # List to store classification predictions and scores for images
    # output_df = pd.DataFrame()

    # print('  # images: %d ' % (len(images)))
    # print('  # labels: %d ' % (len(labels)))
    # print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Load pre-trained model
    # 
    model_load_time = time.time()
    print('\nLoad pre-trained model:')

    # Normalise the image data
    # image_prep = ImagePreprocessing()
    # image_prep.add_featurewise_zero_center()
    # image_prep.add_featurewise_stdnorm()

    # Convolutional network
    # network = input_data(shape=[None, 200, 200, 1], name='input', data_preprocessing=image_prep)
    network = input_data(shape=[None, 200, 200, 1], name='input')
    # network = conv_2d(network, 12, 3, activation='relu', regularizer='L2')
    network = conv_2d(network, 12, 3, activation='relu')
    network = max_pool_2d(network, 2)
    # network = conv_2d(network, 24, 3, activation='relu', regularizer='L2')
    network = conv_2d(network, 24, 3, activation='relu')
    network = max_pool_2d(network, 2)
    # network = conv_2d(network, 36, 3, activation='relu', regularizer='L2')
    network = conv_2d(network, 36, 3, activation='relu')
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
    # 2. Classify images
    # 
    classify_time = time.time()
    print('\nClassify images:')


    predict(IMAGES_FOLDER, OUTPUT_FOLDER, model, 0.9)

    # print('  predictions:')
    # for index, value in class_counts_df.iteritems():
    #     print ('    %s: %d' % (index, value))
    print('  time taken: %.3f seconds' % (time.time() - classify_time))


    # # 
    # # 3. Save prediction results
    # # 

    # # Reorder the DataFrame columns
    # output_df = output_df[['image', 'label', 'prediction', 'confidence']]

    # # Write output to results file
    # output_df.to_csv(s.RESULTS_FILE, index=False)

    # 
    # 4. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    print('Generated file:')
    # print('  %s\n' % (s.RESULTS_FILE))


def tile(filename, w, h):
    i = 0
    j = 0
    w_x = 0
    w_y = 0
    image_tiles = []
    image = misc.imread(filename)

    X = image.shape[0]
    Y = image.shape[1]
    if ((Y > 1228) or (X > 1840)):
        image = misc.imresize(image, [1228, 1840])
    
    # Divide vertically into Y/h tiles
    while (j < int(Y / h)):  
        j += 1

        if (j == int(Y / h)):
            w_y = (Y - h)
        i = 0
        w_x = 0

        # Divide horizontally into X/w tiles
        while (i < int(X / w)): 
            i += 1

            if (i == int(X / w)):
                w_x = (X - w)

            image = image.astype('float32')
            image_tiles.append(image[w_x:w_x + w, w_y:w_y + h])

            w_x += w
        w_y += h
    return image_tiles


def predict(images_folder, output_folder, model, tolerance):
    ''' Returns 1 or 0 based on if transient object is present. If 1, out_filename_tilename is created '''
    outf = {}

    for files in os.listdir(images_folder):
        SET = tile(images_folder + '/' + files, 200, 200)
        flag = 0

        for i in range(len(SET)): 
            result = model.predict([np.reshape(SET[i], [200, 200, 1])])[0][1]

            if (result > tolerance):
                misc.imsave(output_folder + '/out_' + files + '_' + str(i) + '.jpg', SET[i])
                flag = 1

        if flag == 1:
            print (files + '--->1')
        else:
            print (files + '--->0')


if __name__ == '__main__':
    main()
