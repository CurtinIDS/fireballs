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

from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing

# Incrementally train from an existing model
LOAD_EXISTING_MODEL = False
EXISTING_MODEL = s.MODELS_FOLDER + 'experiment/exp5'
# New trained model name
MODEL_NAME = 'synthetic'
# Store model checkpoint files
CHECKPOINT_FOLDER = s.OUTPUT_FOLDER + MODEL_NAME


def main(): 
    # Initialisation
    start_time = time.time() 


    # 
    # 1. Load and prepare the dataset 
    # 
    data_load_time = time.time()
    print('\nLoad and prepare dataset:')

    # Load the training and validation datasets
    X, y = load_images(s.TRAINING_FOLDER)
    X_val, y_val = load_images(s.VALIDATION_FOLDER)
    
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
    network = conv_2d(network, 12, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 24, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 36, 3, activation='relu', regularizer='L2')
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
    model = tflearn.DNN(network, checkpoint_path=CHECKPOINT_FOLDER + '/', tensorboard_dir='../output/logs/')

    # Load a previous model 
    if LOAD_EXISTING_MODEL:
        model.load(EXISTING_MODEL)

    # Train the model
    model.fit({'input': X}, {'target': y}, validation_set=({'input': X_val}, {'target': y_val}), n_epoch=150, batch_size=30, snapshot_epoch=True, show_metric=True)

    print('  time taken: %.3f seconds' % (time.time() - training_start_time))

    # 
    # 4. Display messages to the console
    #
    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


def load_images(path):
    ''' Load dataset images and their labels '''
    images = []
    labels = []

    for subdir in os.listdir(path):
        for file in os.listdir(path + '/' + subdir):
            # Create a one-hot encoding matrix for the image labels based on the sub directory name
            # e.g. /0/ = [1. 0] and /1/ = [0  1.]
            c = np.zeros(2)
            c[int(subdir)] = 1

            # print (file, c)
            labels.append(c)

            image = misc.imread(path + '/' + subdir + '/' + file)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            # Convert the numpy array to float32 to work with image preparation functions
            image = image.astype('float32')
            images.append(image)
    
    return images, labels


if __name__ == '__main__':
    main()
