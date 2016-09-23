"""
Classify astronomy images using the retrained model to detect fireballs

"""
from __future__ import print_function

import os
import fnmatch
import time
import pandas as pd
import settings as s
import tensorflow as tf


def main():

    start_time = time.time()
    
    # 
    # 1. Load the dataset
    # 
    data_load_time = time.time()
    print('\nLoad data:')

    # Retrieve file locations of images in the dataset
    images = [os.path.join(dirpath, f) 
              for dirpath, dirnames, files in os.walk(s.IMAGES_DIRECTORY)
              for f in fnmatch.filter(files, '*.jpg')]

    # Retrieve the class labels
    labels = [line.rstrip() for line in tf.gfile.GFile(s.IMAGE_LABELS_FILE)]

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

    # Unpersist model graph from file
    with tf.gfile.FastGFile(s.MODEL_GRAPH_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    print('  file: %s' % (s.MODEL_GRAPH_FILE))
    print('  time taken: %.3f seconds' % (time.time() - model_load_time))


    # 
    # 2. Classify images
    # 
    classify_time = time.time()
    print('\nClassify images:')

    # Start a TensorFlow session
    with tf.Session() as sess:

        # Retrieve the softmax tensor to generate class predictions
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for image in images:    
            # Read in the image
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            # Feed the image_data as input to the graph and get predictions
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            # predictions = np.squeeze(predictions)
        
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # Extract the image info, top predicted class and confidence score
            label = top_k[0]
            image_file = image.split('/')[4]
            actual_class = image.split('/')[3]
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
    # 3. Save prediction results
    # 

    # Reorder the DataFrame columns
    output_df = output_df[['image', 'label', 'prediction', 'confidence']]

    # Write output to results file
    output_df.to_csv(s.RESULTS_FILE, index=False)

    # 
    # 4. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))

    print('Generated file:')
    print('  %s\n' % (s.RESULTS_FILE))


if __name__ == '__main__':
    main()
