from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow as tf; print(tf.__version__)

# trying to make code from this site work: https://www.tensorflow.org/beta/tutorials/load_data/csv -ERL



# set the path with this; will need change for your purposes -ERL 
train_file_path = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/TUDelft/fastStorage/2013-8/1.csv"
test_file_path = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/TUDelft/fastStorage/2013-8/3.csv"
#csv_path_train = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/dataToManipulate/1_beenManiped.csv"

#look 
np.set_printoptions(precision=3, suppress=True)
#!head {train_file_path} # can find no documnentation to support this statement, but does not seem to work outside of the Google colab environment -ERL


# CSV columns in the input file.
with open(train_file_path, 'r') as f:
    names_row = f.readline()
CSV_COLUMNS = names_row.rstrip('\n').split(';\t')
print(CSV_COLUMNS)

# skipped 2 blocks of code related to adding your own headings (maybe less now) -ERL
# not sure the below three lines does anything relevant -ERL
LABELS = [0, 1]
LABEL_COLUMN = 'Timestamp [ms]'
FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12,
      label_name=None, # setting label_name to None has been the only way I have gotten past this error message -ERL
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

# decided to comment this section out; was taking up too much time and was only meant to be an example printing -ERL
# however will need fixed as calling examples is necessary later -ERL
#examples, labels = next(iter(raw_train_data)) # Just the first batch.
examples = next(iter(raw_train_data))
labels = next(iter(raw_train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

# skipped the lines for preprocessing categorical data as no such data exists in this data set -ERL

# section below is meant to preprocess continuous data to values between 0 and 1 -ERL
def process_continuous_data(data, mean):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

# need to compute the mean for each column of data input; example from site just gives mean value -ERL
# impractical for amount of data we have -ERL

#tf.reduce_mean(raw_train_data) # possible method to deduce mean value; need to call specific tensor I think -ERL

