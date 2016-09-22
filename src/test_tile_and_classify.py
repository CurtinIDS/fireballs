"""
Testing script to construct tiles from a full image and to load
a retrained model to classify these tiles.

"""
from __future__ import print_function

import os
import fnmatch
import time
import settings as s
import tensorflow as tf
import pandas as pd

from skimage import io


def main():

    start_time = time.time()
    
    FILENAME = 'test.jpg'
    TILE_DIRECTORY = s.CACHE_DIRECTORY + 'test/'

    # 
    # 1. Load the dataset
    # 
    data_load_time = time.time()
    print('\nLoad data:')

    # Load the image
    image_data = io.imread(FILENAME)

    print('  time taken: %.3f seconds' % (time.time() - data_load_time))

    # 
    # 2. Generate image tiles
    # 

    image_tiling_time = time.time()
    print('\nGenerate image tiles:')

    # Generate 10 rows x 10 columns of tiles for images
    rows = 10
    cols = 10

    height = int(image_data.shape[0] / rows)
    width = int(image_data.shape[1] / cols)
    
    # Filename info used saving tile images
    filename_parts = FILENAME.split('.') 

    # Generate tiles for each row and column 
    for row in range(rows):
        for col in range(cols):

            # Pixel coordinates for the tile
            x0 = col * width
            x1 = x0 + width
            y0 = row * height
            y1 = y0 + height

            # Generate the tile filename 
            tile_filename = TILE_DIRECTORY + filename_parts[0] + '_' + str(row) + str(col) + '.' + filename_parts[1] 

            # Save the tile image to disk
            io.imsave(tile_filename, image_data[y0:y1, x0:x1, :])

    print('  time taken: %.3f seconds' % (time.time() - image_tiling_time))


    # 
    # 3. Load pre-trained model
    # 
    model_load_time = time.time()
    print('\nLoad pre-trained model:')

    # Unpersist model graph from file
    with tf.gfile.FastGFile(s.MODEL_GRAPH_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    print('  file: %s' % (s.MODEL_GRAPH_FILE))
    print('  time taken: %.3f seconds' % (time.time() - model_load_time))


    # 
    # 4. Classify images
    # 
    classify_time = time.time()
    print('\nClassify images:')

    images = [os.path.join(dirpath, f) 
              for dirpath, dirnames, files in os.walk(TILE_DIRECTORY)
              for f in fnmatch.filter(files, '*.jpg')]

    # Retrieve the class labels
    labels = [line.rstrip() for line in tf.gfile.GFile(s.IMAGE_LABELS_FILE)]

    # List to store classification predictions and scores for images
    output_df = pd.DataFrame()

    # Start a TensorFlow session
    with tf.Session() as sess:

        # Retrieve the softmax tensor to generate class predictions
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for image in images:    
            # Read in the image
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            # Feed the image_data as input to the graph and get predictions
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # Append the output list with the top predicted class and confidence score
            # Extract the image info, top predicted class and confidence score
            label = top_k[0]
            image_file = image.split('/')[3]
            actual_class = image.split('/')[2]
            label_name = labels[label]
            score = '%.5f' % (predictions[0][label])
            
            # Append the prediction to the output DataFrame
            output_df = output_df.append({
                'image': image_file, 
                'label': actual_class, 
                'prediction': label_name, 
                'confidence': score}, 
                ignore_index=True)      

    # Calculate the number of images classified into each class
    class_counts_df = output_df.groupby('prediction')['prediction'].count()

    print('  predictions:')
    for index, value in class_counts_df.iteritems():
        print ('    %s: %d' % (index, value))
    print('  time taken: %.3f seconds' % (time.time() - classify_time))


    # 
    # 5. Display prediction results
    # 

    # Reorder the DataFrame columns
    output_df = output_df[['image', 'label', 'prediction', 'confidence']]

    # Only display info of tiles predicted to contain meteors
    print(output_df[output_df['prediction'] == 'meteors'])

    # Write output to results file
    # output_df.to_csv(s.RESULTS_FILE, index=False)

    # 
    # 6. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    print('Generated file:')
    print('  %s\n' % (s.RESULTS_FILE))


if __name__ == '__main__':
    main()
