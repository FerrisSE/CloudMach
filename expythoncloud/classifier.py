from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
print("starting Cloud PRofiler")
print(tf.__version__)

import tensorflow_datasets as tfds

# Working through the TF Load CSV tutorial -ERL
# link: https://www.tensorflow.org/beta/tutorials/load_data/csv -ERL


train_file_path = 'NoWSp.csv'
# alternatively , accessing a dataset through a URL 
#TRAIN_DATA_URL="train.csv"
#TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
#train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

test_file_path = "3.csv"



np.set_printoptions(precision=3, suppress=True)



# CSV columns in the input file.
with open(train_file_path, 'r') as f:
    names_row = f.readline()


CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

LABELS = [0, 1]
LABEL_COLUMN = 'Timestampms'

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
raw_test_data = get_dataset(test_file_path)


examples, labels = next(iter(raw_train_data)) # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

# the next section in the tutorial walks you through preprocessing categorical data; not relevant to out data set -ERL
# below that is a section titled 'continuous data' I am going to attempt to implement that below this line -ERL

def process_continuous_data(data, mean):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

MEANS = {
    'CPUcores' : 4,
    'CPUcapacityprovisionedMHZ' : 11704,
    'CPUusageMHZ' : 470.808806738965,
    'CPUusagePerc' : 4.02263224142337,
    'MemorycapacityprovisionedKB' : 67100000,
    'MemoryusageKB' : 396850.70392986,
    'DiskreadthroughputKB/s' : 9.06871275576744,
    'DiskwritethroughputKB/s' : 412.470581511162,
    'NetworkreceivedthroughputKB/s' : 12.5482411726785,
    'NetworktransmittedthroughputKB/s' : 2.98265049762985
    #'age' : 29.631308,
    #'n_siblings_spouses' : 0.545455,
    #'parch' : 0.379585,
    #'fare' : 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
  numerical_columns.append(num_col)
  
print(numerical_columns)

preprocessing_layer = tf.keras.layers.DenseFeatures(numerical_columns)

# 'Build the model' section of tutorial

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


# 'Train, evaluate, and predict' section of tutorial

train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

model.fit(train_data, epochs=20)

# EVERYTHING BELOW THIS LINE ARE THE PARTS I SUGGESTED MAY HAVE CAME FROM A DIFFERENT SOURCE -ERL
##########################################################
# a little confused where the below lines came from; I know I entered them, but they do not seem to be a part of the tutorial as far as I can tell 7/19 -ERL
# I am thinking they have been updating/altering their tutorial as they develop TF 2.0
# That would explain some of the inconsistencies -ERL

#MC_Provisioned_tensor = examples['Memorycapacityprovisioned[KB]']
#print(MC_Provisioned_tensor) # had to add print to show what is expected from the example -ERL

#print(process_continuous_data(MC_Provisioned_tensor, MEANS['Memorycapacityprovisioned[KB]'])) # print added to show expected values -ERL
# values all process to 0.5
# makes sense as all values in this column are indentical
# calculated average only differs due to three corrupted lines in dataset -ERL

def preprocess(features, labels):
  
  # removed catergorical preprocessing function -ERL
  
  # Process continuous features.
  for feature in MEANS.keys():
    features[feature] = process_continuous_data(features[feature],
                                                MEANS[feature])
  
  # Assemble features into a single tensor.
  features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)
  
  return features, labels

#train_data = raw_train_data.map(preprocess).shuffle(500)
#test_data = raw_test_data.map(preprocess)

#examples, labels = next(iter(train_data))

#examples, labels
