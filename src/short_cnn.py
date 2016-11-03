#
# Instructions
# -touch a file "checkpoint"
# -copy models/synthetic to root
# -resize to 25% of the orinigal image
# -change predict function towards the end to include test director
#
#
import tflearn
import os
from timeit import timeit
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.layers.normalization import local_response_normalization
from random import random
from scipy import misc


def main(): 
    # A,B=loadimage('training')
    # C,D=loadimage('temp')
    # C,D=loadimage('validation')


    network = input_data(shape=[None, 200, 200, 1], name='input')  # 200x200 tile
    network = cnn(network, 3)
    network = fully_connected(network, 10, activation='tanh')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, checkpoint_path='.ckpt')
    if os.path.exists('checkpoint'):  # if checkpoint file found, then load the model
        model.load("trial")


    # model.fit({'input':A},{'target': B},validation_set=({'input':C},{'target':D}),n_epoch=150,batch_size=30,snapshot_epoch=True,show_metric=True)

    # start=timeit()

    predict('xiang', model, 0.9)
    # print (timeit()-start)


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

    X=im.size[0]
    Y=im.size[1]
    if ((X>1840) or (Y>1228)):
        im=im.thumbnail(1840,1228)
    
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


# Returns 1/0 based on if transient object is preset. If 1, out_filename_tilename would be created
def predict(directory, model, tolerance):
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
