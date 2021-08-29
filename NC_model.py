import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
import vggish_params as params

def build_model(x_width, x_height, x_channels):
  model = Sequential()

  model.add(Conv2D(16, kernel_size=(4, 4),
            input_shape=(x_width, x_height, x_channels), name='Conv2D0_16_4_4'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.3))

  model.add(Conv2D(32, kernel_size=(4, 4), name='Conv2D1_32_4_4'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.3))

  model.add( Conv2D(16, kernel_size=(4, 4), name='Conv2D2_16_4_4') )
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(params.NUM_CLASS, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

  return model
