import math
import os

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras

import data_prepare
import network

model_dir = './models'
model_path = './models/model.h5'
def mean_squared_error(y_true, y_pred):
    print(y_true,y_pred)
    return K.mean((y_true-y_pred)**2)
def peak_sifnal_to_noise(y_true, y_pred):
    return 10*keras.backend.log(1/mean_squared_error(y_true, y_pred))/math.log(10)

def categorical_crossentropy(y_true, y_pred):
    y_pred = keras.backend.clip(y_pred, 1e-6, 1 - 1e-6)
    return keras.losses.categorical_crossentropy(y_true,y_pred)

def get_model_from_load():
    return keras.models.load_model(model_path,compile=False)

def get_model_from_network(channel):
    return network.unet(channel)


if __name__=='__main__':
    x,y_label = data_prepare.get_train_data()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(model_path):
        print("get_model_from_load")
        model = get_model_from_load()
        learning_rate = 0.0001
    else:
        print("get_model_from_network")
        model = get_model_from_network(y_label.shape[-1])
        learning_rate = 0.001

    model.summary()
    checkpointer = keras.callbacks.ModelCheckpoint('./models/model_{epoch:03d}.hdf5',
                                                   verbose=1, save_weights_only=False, period=10)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=categorical_crossentropy)
    history = model.fit(
        x=x,
        y=keras.utils.to_categorical(y_label),
        batch_size= 128,
        epochs=1,
        validation_split=0.1,
        callbacks=[checkpointer]
    )
    model.save("./models/model.h5")