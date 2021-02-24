import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import data_prepare

model_path = "./models/model.h5"

if __name__ == "__main__":
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        x = data_prepare.get_test_data()

        for i in range(len(x)):
            y = model.predict(x[i][np.newaxis, ..., np.newaxis])
            print(y.shape, x[i].shape)
            plt.subplot(121), plt.imshow(x[i],cmap='gray'), plt.title('input')
            plt.subplot(122), plt.imshow(y[0, ::, ::, 0],cmap='gray'), plt.title('output')
            plt.show()
