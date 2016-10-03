"""
Summarise classification results on the fireball detection dataset

"""
from __future__ import print_function

import time
import pandas as pd
import settings as s


def main():  
    # Initialisation
    start_time = time.time()


    # 
    # 1. Load the dataset
    # 

    data_load_time = time.time()
    print('\nLoad results:')

    # load the data file
    df = pd.read_csv(s.RESULTS_FILE, index_col=False)

    print('  # records: %d ' % (len(df)))
    print('  time taken: %.3f seconds' % (time.time() - data_load_time))


    # 
    # 2. Summarise experimental results
    # 

    data_prep_time = time.time()
    print('\nSummarise results:')

    # Initialise variables
    label_column = 'label'
    # Specify transient objects as the positive (first) class
    labels = s.LABELS
    # labels = df['label'].unique()
    classes_data = []

    # Create dataframes containing records for each class label
    # This summary only works for two classes / binary classification
    for index, label in enumerate(labels):
        classes_data.append(df[df[label_column] == label])

    # Extract measures for calculating performance metrics
    tp = len(classes_data[0][classes_data[0]['prediction'] == labels[0]])
    fp = len(classes_data[1][classes_data[1]['prediction'] == labels[0]])
    tn = len(classes_data[1][classes_data[1]['prediction'] == labels[1]])
    fn = len(classes_data[0][classes_data[0]['prediction'] == labels[1]])

    # Calculate performance metrics
    accuracy = (float(tp) + tn) / (tp + fp + tn + fn)
    precision = (float(tp) / (tp + fp))
    recall = (float(tp) / (tp + fn))
    f1_score = 2 * (precision * recall / (precision + recall))

    print('  Predictions:')

    # Print out the class labels and the counts
    for index, label in enumerate(labels):
        # Determine the correct label type
        label_type = 'positive'
        if index:
            label_type = 'negative'

        print('    %s (%s): %d' % (label, label_type, len(classes_data[index])))

    print('  Confusion matrix:')
    print('    true positives: %d' % (tp))
    print('    false positives: %d' % (fp))
    print('    true negatives: %d' % (tn))
    print('    false negatives: %d' % (fn))
    print('  Metrics:')
    print('    accuracy: %.3f (%d/%d: %d incorrect)' % (accuracy, tp + tn, tp + fp + tn + fn, fp + fn))
    print('    precision: %.3f' % (precision))
    print('    recall: %.3f' % (recall))
    print('    f1 score: %.3f' % (f1_score))    
    print('  time taken: %.3f seconds' % (time.time() - data_prep_time))


    # 
    # 3. Display messages to the console
    #

    print('\nTotal time: %.3f seconds\n' % (time.time() - start_time))


if __name__ == '__main__':
    main()
