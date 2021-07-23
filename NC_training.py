# This file should be executed from parent dir, as:
#   "~$: python ./NC/NC_training.py"

# Imports -------------------------------------------------
import librosa
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
# Imports DSP module
if os.getcwd().endswith(('\\NC', '/NC')): os.chdir('..')
sys.path.insert(1, os.getcwd())
import vggish_input as vi
import vggish_params as params
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Activation, Flatten, BatchNormalization

def load_prepro_noise_dataset(dataset_paths: dict, batch_size=50, verbose=False):
  '''
    Load the noise dataset and preprocess. Return 2D specturms and one-hot labels.
    Load the noise dataset from disk, resample to params.SAMPLE_RATE, padding to SAMPLE_LEN
    and convert to mel spectrum.

    Parameters
    ----------
    dataset_paths: A dict has following structure
      {
        "classA": {"filenames": [path_0, path_1, ...]},
        "classB": {"filenames": [path_0, path_1, ...]},
        ...
      }

    batch_size: The number of files to load from disk at once.

    verbose: bool, print more info or not

    Return
    ------
    (dataset_x, dataset_y):
      _x.shape = (file_counts, mels, time_slices)
      _y.shape = (file_counts, categories) one-hot encoding
  '''

  total_files = 0
  processed_files = 0  # Eventually, processed will equals to total
  padded_files = 0     # padded wav counts
  resampled_files = 0  # resampled wav counts
  dataset_x = []  # Time domains to each wav
  dataset_y = []  # labels, one-hot encoded

  # Get total filecounts
  for cats in dataset_paths:
    total_files += len(dataset_paths[cats]['filenames'])

  # Go through each category
  for c, cats in enumerate(dataset_paths):
    filenames = dataset_paths[cats]['filenames']
    cat_files = len(filenames) # file counts to this category
    f = 0

    # Go through each file in a category
    while f < cat_files:
      t_ = [] # List of time domain samples, 1 element to samples of 1 file
      f_index = f
      t_index = 0
      
      # Print progress
      print(f'\rLoading and FFTing category {c+1}/{params.NUM_CLASS}, ', end='')
      print(f"processed file {processed_files+f+1}/{total_files}...   ", end='')
      
      # Batch load wav from disk
      while f_index-f < batch_size and f_index < cat_files:
        # Read time domain samples and sample rate from wav
        # Only resmaple if the sample rate of file is not as specified
        sample, sr = librosa.load(filenames[f_index], sr=None)
        if sr != params.SAMPLE_RATE:
          if verbose:   print(f'{filenames[f_index]} is resmapled from {sr} to {params.SAMPLE_RATE}.')
          resampled_files += 1
          sample = librosa.resample(sample, orig_sr=sr, target_sr=params.SAMPLE_RATE, res_type='polyphase')
        t_.append(sample)
        f_index += 1
      
      # Batch preprocess
      while t_index < batch_size and t_index < f_index-f:
        # Convert to spectrum
        arr = vi.waveform_to_examples(t_[t_index], params.SAMPLE_RATE)
        # arr[num_spectrums, num_frames, num_bands] 
        dataset_x.extend(arr)
        # Convert to one hot
        one_hot_en = to_categorical(c, params.NUM_CLASS)
        dataset_y.extend([one_hot_en]*arr.shape[0])
        t_index = t_index + 1
      
      f += batch_size

    # Update progress for printing
    processed_files += cat_files

  # Convert to numpy
  dataset_x = np.array(dataset_x)
  dataset_y = np.array(dataset_y)

  print(f'\rDataset loaded and FFTed. '
        f'Total samples:{processed_files}, '
        f'resampled:{resampled_files}, '
        f'padded:{padded_files}.')
  return (dataset_x, dataset_y)

def load_time_domains(dataset_paths: dict,  batch_size=50, verbose=False):
  '''
    Load the noise dataset from disk, resample to params.SAMPLE_RATE, padding to SAMPLE_LEN
    and return 1-D array of time samples and labels.

    Parameters
    ----------
    dataset_paths: A dict has following structure
      {
        "classA": {"filenames": [path_0, path_1, ...]},
        "classB": {"filenames": [path_0, path_1, ...]},
        ...
      }

    batch_size: The number of files to load from disk at once.
                 The loop is: load batch_size of wav files, resample and padding.

    verbose: bool, print more info or not
  '''
  
  total_files = 0
  processed_files = 0  # Eventually, processed will equals to total
  padded_files = 0     # padded wav counts
  resampled_files = 0  # resampled wav counts
  dataset_x = []       # Time domains to each wav
  dataset_y = []       # labels, one-hot encoded

  # Get total filecounts
  for cats in dataset_paths:
    total_files += len(dataset_paths[cats]['filenames'])

 # Go through each category
  for c, cats in enumerate(dataset_paths):
    filenames = dataset_paths[cats]['filenames']
    cat_files = len(filenames) # file counts to this category
    f = 0

    # Go through each file in a category
    while f < cat_files:
      # List of time domain samples and sample rate.
      t_ = []      # t_[i][k]: i for i-th file, k=0 for time samples, k=1 for sample rate
      f_index = f
      t_index = 0
      
      # Print progress
      print(f'\rLoading and FFTing category {c+1}/{params.NUM_CLASS}, ', end='')
      print(f"processed file {processed_files+f+1}/{total_files}...   ", end='')
      
      # Batch load wav from disk
      while f_index-f < batch_size and f_index < cat_files:
        # Read time domain samples and sample rate from wav
        # Only resmaple if the sample rate of file is not as specified
        sample, sr = librosa.load(filenames[f_index], sr=None)
        t_.append([sample, sr])
        f_index += 1
      
      # Resample and padding
      while t_index < batch_size and t_index < f_index-f:
        sample = t_[t_index][0]
        sr     = t_[t_index][1]
        
        # Resample
        if sr != params.SAMPLE_RATE:
          if verbose:   print(f'{filenames[f+t_index]} is resmapled from {sr} to {params.SAMPLE_RATE}.')
          resampled_files += 1
          sample = librosa.resample(sample, orig_sr=sr, target_sr=params.SAMPLE_RATE)
        # Pad sample to 10 seconds
        if sample.shape[0] < SAMPLE_LEN:
          if verbose:   print(f'{filenames[f+t_index]} is padded from {sample.shape[0]} to {SAMPLE_LEN} samples.')
          padded_files += 1
          paddings = SAMPLE_LEN - sample.shape[0]
          sample = np.pad(sample, pad_width=[0, paddings], mode='constant')
        
        dataset_x.append(sample)
        # Convert to one hot
        one_hot_en = np.eye(params.NUM_CLASS)[c]
        dataset_y.append(one_hot_en)
        t_index += 1
      
      f += batch_size

    # Update progress for printing
    processed_files += cat_files

  dataset_x = np.array(dataset_x)
  dataset_y = np.array(dataset_y)
  print(f'\rDataset loaded and FFTed. '
        f'Total samples:{processed_files}, '
        f'resampled:{resampled_files}, '
        f'padded:{padded_files}.')
  return (dataset_x, dataset_y)

# Load and Proprocess the dataset -------------------------
with open(params.PATH_NOISE_LIST) as json_file:
  dataset_paths = json.load(json_file)

train_x, train_y = load_prepro_noise_dataset(dataset_paths['train'], batch_size=300, verbose=True)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1).astype('float32')
from sklearn.utils import shuffle
train_x, train_y = shuffle(train_x, train_y)

# This model overfits, I think it's the problem of the dataset and preprocess
# yamNet gives a good result with AudioSet. Go check it out for its preproessing and model.
# TODO: Instead of training with all 512 categories, 
#       narrow down the categories taken from AudioSet, to 4 for example.
# Build model ---------------------------------------------
BATCH_SIZE = 50
EPOCHS = 5

model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5), # 32 filters of 3x3
          input_shape=(train_x.shape[1], train_x.shape[2], 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
Dropout(0.3)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(4, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
Dropout(0.3)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(params.NUM_CLASS, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# Training
# history_CNN = model.fit(x=train_x, y=train_y, validation_split=0.2,
#                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# model.save_weights('model_0719.h5')
model.load_weights('model_0719.h5')
model.call(vi.wavfile_to_examples())
