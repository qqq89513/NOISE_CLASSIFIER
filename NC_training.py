# Imports -------------------------------------------------
import librosa
import os, sys, gc, json
import numpy as np
import matplotlib.pyplot as plt
import vggish_input as vi
import vggish_params as params
from utils.handle_dataset import load_prepro_noise_dataset
from utils.handle_dataset import make_cats_equal

with open(params.PATH_NOISE_LIST) as json_file:
  dataset_paths = json.load(json_file)

# Load and Proprocess the dataset -------------------------
train_x, train_y = load_prepro_noise_dataset(
    dataset_paths['train'], equal_samples=True, shuffle=True, reshape_x=True, verbose=True)
eval_x, eval_y = load_prepro_noise_dataset(
    dataset_paths['test'], equal_samples=True, shuffle=True, reshape_x=True, verbose=True)

gc.collect() # Garbage collection

# This model overfits, I think it's the problem of the dataset and preprocess
# yamNet gives a good result with AudioSet. Go check it out for its preproessing and model.
# TODO: Instead of training with all 512 categories, 
#       narrow down the categories taken from AudioSet, to 4 for example.
# Build model ---------------------------------------------
BATCH_SIZE = 50
EPOCHS = 10

from tensorflow.keras.callbacks import EarlyStopping
import NC_model

# Load model
model = NC_model.build_model(train_x.shape[1], train_x.shape[2], train_x.shape[3])

# Training
ES_callback = EarlyStopping(monitor='val_loss', patience=4, baseline=5, restore_best_weights=True)
history_CNN = model.fit(
  x=train_x, y=train_y,
  validation_data=(eval_x, eval_y),
  callbacks=ES_callback,
  epochs=EPOCHS, batch_size=BATCH_SIZE,
  verbose=1)

# model.save_weights('model_0724_3_cats_equal_samples.h5')
# model.load_weights('model_0724_3_cats_equal_samples.h5')

model.save('model_0928_3_cats_equal_samples.h5')

class_names = list(dataset_paths['train'].keys())

# Inference example
predicted = np.array(model(eval_x[230:233]))
print(f"predicted={predicted}")
print(f"true ground={eval_y[230:233]}")
