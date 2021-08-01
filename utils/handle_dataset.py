
import numpy as np
import vggish_input as vi
import vggish_params as params
from sklearn.utils import shuffle as shf
import librosa

def to_categorical_1D(y: int, num_classes: int):
  if y >= num_classes:
    raise ValueError(f"y should be in range [0, num_classes), " 
      f"Given y={y} and num_classes={num_classes}.")
  labels = np.zeros(num_classes)
  labels[y] = 1
  return labels

def make_cats_equal(x, y, num=None):
  num_of_each_cats = np.sum(y, axis=0).astype(np.int64)
  min_cats_num = np.min(num_of_each_cats)
  if num is int and (num > min_cats_num or num < 1):
    raise ValueError(
      f"num should be in range [1, # of samples of the smallest class], "
      f"which is [1, {min_cats_num}]")

  ret_x = []
  ret_y = []
  for i in range(params.NUM_CLASS):
    label = to_categorical_1D(i, params.NUM_CLASS)
    match_to_cat = np.all(y==label, axis=-1)
    indices_to_be_pick = np.where(match_to_cat)[0]
    # Generate random index
    picked = np.random.choice(
      a=indices_to_be_pick, # random pick size items among a
      size=min_cats_num,    # picked.shape=size
      replace=False)        # Non repeated
    ret_x.extend(x[picked])
    ret_y.extend(y[picked])
  
  ret_x = np.array(ret_x)
  ret_y = np.array(ret_y)
  return ret_x, ret_y

def num_to_class(num: int, dataset_paths) -> str:
  if num < 0 or num >= params.NUM_CLASS:
    raise ValueError(
      f"num should be in range [{0}, params.NUM_CLASS], "
      f"where params.NUM_CLASS={params.NUM_CLASS}")
  total_cats = list(dataset_paths['dataset'].keys())
  return total_cats[num]

def load_prepro_noise_dataset(dataset_paths: dict, batch_size=50, equal_samples=True, shuffle=True, reshape_x=True, verbose=False):
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
        one_hot_en = to_categorical_1D(c, params.NUM_CLASS)
        dataset_y.extend([one_hot_en]*arr.shape[0])
        t_index = t_index + 1
      
      f += batch_size

    # Update progress for printing
    processed_files += cat_files

  # Convert to numpy
  dataset_x = np.array(dataset_x)
  dataset_y = np.array(dataset_y)

  print(f'\rDataset loaded and FFTed. '
        f'Total samples:{dataset_x.shape[0]}. '
        f'Total wav files:{processed_files}, '
        f'resampled:{resampled_files}, '
        f'padded:{padded_files}.')

  if verbose:
    print(f"x.shape={dataset_x.shape}")
    if equal_samples:   print(f"Original samples to different classes={np.sum(dataset_y, axis=0)}")
  if equal_samples:     dataset_x, dataset_y = make_cats_equal(dataset_x, dataset_y)
  if verbose:           print(f"Samples to different classes={np.sum(dataset_y, axis=0)}")
    
  if shuffle:
    dataset_x, dataset_y = shf(dataset_x, dataset_y)


  if reshape_x:
    dataset_x = dataset_x.reshape(dataset_x.shape[0], dataset_x.shape[1], dataset_x.shape[2], 1).astype('float32')

  return (dataset_x, dataset_y)
