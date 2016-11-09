#
# Instructions
# -touch a file "checkpoint"
# -copy models/synthetic to root
# -resize to 25% of the orinigal image
# -change predict function towards the end to include test director
#
#

import os
import numpy as np
import tensorflow as tf
import tflearn

from random import random
from scipy import misc

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

MODEL_FILE = '../models/synthetic'
# DATA_FOLDER = '../cache/xiang'
TRAINING_DATA_FOLDER = '../cache/post'
VALIDATION_DATA_FOLDER = '../cache/validation'
DATA_FOLDER = '../cache/new'



def main(): 
    # A, B = loadimage(TRAINING_DATA_FOLDER)
    # C, D = loadimage(VALIDATION_DATA_FOLDER)

    # Network architecture

    # Input data - 200 x 200 grayscale tile
    network = input_data(shape=[None, 200, 200, 1], name='input')

    # 3 convolution and max pooling layers
    network = cnn(network, 3)

    network = fully_connected(network, 10, activation='tanh')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, checkpoint_path='.ckpt')

    if os.path.exists('checkpoint'):  # if checkpoint file found, then load the model
        model.load(MODEL_FILE)

    # model.fit(
    #     {'input':A},
    #     {'target': B},
    #     validation_set=({'input':C},{'target':D}),n_epoch=150,batch_size=30,snapshot_epoch=True,show_metric=True)

    predict(DATA_FOLDER, model, 0.9)


def cnn(network, count):
    for i in range(count):
        network = conv_2d(network, (i + 1) * 12, 3, activation='relu')
        network = max_pool_2d(network, 2)
        # network=local_response_normalization(network)
    return (network)


def loadimage(path):
    A = []
    B = []

    for subdir in os.listdir(path):
        for files in os.listdir(path + '/' + subdir):
            # print (files)
            c = np.zeros(2)
            c[int(subdir)] = 1
            B.append(c)
            im = misc.imread(path + '/' + subdir + '/' + files)
            im = np.reshape(im, [im.shape[0], im.shape[1], 1])
            A.append(im)

    return A, B


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
    ''' Returns 1/0 based on if transient object is preset. If 1, out_filename_tilename would be created '''
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
