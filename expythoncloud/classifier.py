from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
print("starting Cloud PRofiler")
print(tf.__version__)

import tensorflow_datasets as tfds

# set the path with this; will need change for your purposes -ERL 
#train_file_path = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/TUDelft/fastStorage/2013-8/1.csv"
#test_file_path = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/TUDelft/fastStorage/2013-8/3.csv"
#csv_path_train = "C:/Users/Mr. E/Documents/My Documents/School Work/research project/datasets/dataToManipulate/1_beenManiped.csv"

train_file_path = 'NoWSp.csv'
#train_file_path = "train.csv"
# alternatively , accessing a dataset through a URL 
#TRAIN_DATA_URL="train.csv"
#TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
#train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

test_file_path = "3.csv"
csv_path_train = "1_beenManiped.csv"


np.set_printoptions(precision=3, suppress=True)



# CSV columns in the input file.
with open(train_file_path, 'r') as f:
    names_row = f.readline()


CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

LABELS = [0, 1]
LABEL_COLUMN = 'Timestamp[ms]'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]


def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
#raw_test_data = get_dataset(test_file_path)


examples, labels = next(iter(raw_train_data)) # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)