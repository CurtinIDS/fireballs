"""
Train a convolutional neural network for detecting transient objects (streaks)
from optical images of the night sky

"""

import os
import time
import numpy as np
import tensorflow as tf
import tflearn
import settings as s
from random import random
from scipy import misc

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing

MODEL = 'synthetic'
CHECKPOINT_FOLDER = s.OUTPUT_DIRECTORY + MODEL
TRAINING_FOLDER = s.SYNTHETIC_DIRECTORY + 'training'
VALIDATION_FOLDER = s.SYNTHETIC_DIRECTORY + 'validation'


def main(): 
    # Initialisation
    start_time = time.time() 


    # 
    # 1. Load and prepare the dataset 
    # 
    data_load_time = time.time()
    print('\nLoad and prepare dataset:')

    X, y = load_images(TRAINING_FOLDER)
    X_val, y_val = load_images(VALIDATION_FOLDER)

    # Normalise the image data
    image_prep = ImagePreprocessing()
    image_prep.add_featurewise_zero_center()
    image_prep.add_featurewise_stdnorm()

    print('  # training images: %d ' % (len(X)))
    print('  # validation images: %d ' % (len(X_val)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Specify the neural network architecture
    # 
    training_start_time = time.time()
    print('\nSpecify the neural network architecture:')
        
    # Convolutional network
    network = input_data(shape=[None, 200, 200, 1], name='input', data_preprocessing=image_prep)
    # network = input_data(shape=[None, 200, 200, 1], name='input')
    network = conv_2d(network, 12, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 24, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 36, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 10, activation='tanh')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    print('  time taken: %.3f seconds' % (time.time() - training_start_time))


    # 
    # 3. Train the convolutional neural network
    # 
    training_start_time = time.time()
    print('\nTrain the neural network:')    

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Specify the model output folder
    model = tflearn.DNN(network, checkpoint_path=CHECKPOINT_FOLDER + '/')

    # Load a previous model checkpoint 
    # if os.path.exists(CHECKPOINT_FOLDER + '/checkpoint'):
    #     model.load(MODEL_FILE)

    # Train the model
    model.fit({'input': X}, {'target': y}, validation_set=({'input': X_val}, {'target': y_val}), n_epoch=150, batch_size=30, snapshot_epoch=True, show_metric=True)

    print('  time taken: %.3f seconds' % (time.time() - training_start_time))

    # 
    # 4. Display messages to the console
    #
    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    #
    # Instructions
    # -touch a file "checkpoint"
    # -resize to 25% of the original image
    # -change predict function towards the end to include test director
    #
    # MODEL_FILE = '../models/synthetic'
    # TEST_FOLDER = '../cache/new'
    # if checkpoint file found, then load the model
    # model = tflearn.DNN(network, checkpoint_path='.ckpt')
    # if os.path.exists('checkpoint'):
    #     model.load(MODEL_FILE)
    # predict(TEST_FOLDER, model, 0.9)


def load_images(path):
    images = []
    labels = []

    for subdir in os.listdir(path):
        for file in os.listdir(path + '/' + subdir):
            # Create a one-hot encoding matrix for the image labels based on the sub directory name
            # e.g. /0/ = [1. 0] and /1/ = [0  1.]
            c = np.zeros(2)
            c[int(subdir)] = 1
            labels.append(c)

            image = misc.imread(path + '/' + subdir + '/' + file)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            # Convert the numpy array to float32 to work with image preparation functions
            image = image.astype('float32')
            images.append(image)

    return images, labels


def tile(filename, w, h):
    i = 0
    j = 0
    w_x = 0
    w_y = 0
    image_tiles = []
    im = misc.imread(filename)

    X = im.shape[0]
    Y = im.shape[1]
    if ((Y > 1228) or (X > 1840)):
        im = misc.imresize(im, [1228, 1840])
    
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
            image_tiles.append(im[w_x:w_x + w, w_y:w_y + h])

            w_x += w
        w_y += h
    return image_tiles


def predict(directory, model, tolerance):
    ''' Returns 1 or 0 based on if transient object is present. If 1, out_filename_tilename is created '''
    outf = {}

    for files in os.listdir(directory):
        SET = tile(directory + '/' + files, 200, 200)
        flag = 0

        for i in range(len(SET)): 
            result = model.predict([np.reshape(SET[i], [200, 200, 1])])[0][1]

            if (result > tolerance):
                misc.imsave(directory + '/out_' + files + '_' + str(i) + '.jpg', SET[i])
                flag = 1

        if flag == 1:
            print (files + '--->1')
        else:
            print (files + '--->0')


if __name__ == '__main__':
    main()
