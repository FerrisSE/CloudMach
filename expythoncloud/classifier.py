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

# the next section in the tutorial walks you through preprocessing categorical data; not relevant to out data set -ERL
# below that is a section titled 'continuous data' I am going to attempt to implement that below this line -ERL

def process_continuous_data(data, mean):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

MEANS = {
    'CPUcores' : 4.00590279305002,
    'CPUcapacityprovisioned[MHZ]' : 11702.7515990837,
    'CPUusage[MHZ]' : 7715.7873247688,
    'CPUusage[%]' : 4.84809453923082,
    'Memorycapacityprovisioned[KB]' : 67101715.6335748,
    'Memoryusage [KB]' : 452856.5507969,
    'Diskreadthroughput[KB/s]' : 15.2946242650972,
    'Diskwritethroughput[KB/s]' : 501.937240638949,
    'Networkreceivedthroughput[KB/s]' : 20.8283634408275,
    'Networktransmittedthroughput[KB/s]' : 3.07516216605382
    #'age' : 29.631308,
    #'n_siblings_spouses' : 0.545455,
    #'parch' : 0.379585,
    #'fare' : 34.385399
}

MC_Provisioned_tensor = examples['Memorycapacityprovisioned[KB]']
print(MC_Provisioned_tensor) # had to add print to show what is expected from the example -ERL

print(process_continuous_data(MC_Provisioned_tensor, MEANS['Memorycapacityprovisioned[KB]'])) # print added to show expected values -ERL
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

train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)

examples, labels = next(iter(train_data))

examples, labels
