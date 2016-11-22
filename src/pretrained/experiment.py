"""
Script to run the fireballs detection experiment pipeline

"""
from __future__ import print_function

import time
import subprocess
import settings as s



def main():  
    # Initialisation
    start_time = time.time()


    # 
    # 1. Prepare the dataset
    # 

    # print('\n--  Prepare dataset  --')
    # subprocess.call('python prepare_dataset.py', shell=True)

    
    # 
    # 2. Train the classifier and run the experiment
    # 

    print('\n-- Train classifier --')

    train_command = 'python retrain.py'
    train_command += ' --bottleneck_dir=' + s.BOTTLENECK_DIRECTORY
    train_command += ' --how_many_training_steps=' + str(s.TRAINING_STEPS)
    train_command += ' --model_dir=' + s.INCEPTION_DIRECTORY
    train_command += ' --output_graph=' + s.MODEL_GRAPH_FILE
    train_command += ' --output_labels=' + s.IMAGE_LABELS_FILE 
    train_command += ' --image_dir ' + s.IMAGES_DIRECTORY 

    # if s.EXPERIMENT_PREPROCESSING == '_rotated':
    #    train_command += ' --flip_left_right=True '

    print(train_command)

    subprocess.call(train_command, shell=True)


    # 
    # 3. Use the trained classifier to run predictions on the complete dataset
    # 

    # print('\n-- Classify dataset  --')
    # subprocess.call('python retrain_classify_images.py', shell=True)

    # 
    # 4. Summarise the experimental results
    # 

    # print('\n-- Summarise results --')
    # subprocess.call('python summarise_results.py > ' + s.RESULTS_SUMMARY_FILE, shell=True)


    # 
    # 5. Parse the experiment results
    # 

    # print('\n-- Parse results --')
    # subprocess.call('python parse_results.py', shell=True)
   

    # 
    # 6. Display messages to the console
    #

    print('\nExperiment time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
