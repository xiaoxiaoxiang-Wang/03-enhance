import os

import cv2
import numpy as np

import matplotlib.pyplot as plt

train_data_path = "./data/train"
train_label_data_path = "./data/train_cleaned"
test_data_path = "./data/test"
pitch_height = 48
pitch_width = 48
stride = 5
scales = (1, 0.9, 0.8, 0.7)
transform = {
    "origin": lambda img: img,
    "rot90_1":lambda img:np.rot90(img),
    # "rot90_2":lambda img:np.rot90(img,2),
    # "rot90_3":lambda img:np.rot90(img,3),
    # "flip":lambda img:np.flipud(img),
    "flip_rot90_1":lambda img:np.flipud(np.rot90(img)),
    # "flip_rot90_2": lambda img: np.flipud(np.rot90(img,2)),
    # "flip_rot90_3": lambda img: np.flipud(np.rot90(img,3)),
}


def generate_data(input_path, label_path):
    input_files = os.listdir(input_path)
    label_files = os.listdir(label_path)
    x = []
    y = []
    for input_file, label_file in zip(input_files, label_files):
        if input_file.endswith(".db"):
            continue
        input = cv2.imread(filename=os.path.join(input_path, input_file), flags=cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(filename=os.path.join(label_path, label_file), flags=cv2.IMREAD_GRAYSCALE)
        input = input / 255.0
        h, w = input.shape[0:2]
        for scale in scales:
            # 切片左闭右开
            h_scale, w_scale = int(h * scale), int(w * scale)
            input_scale = cv2.resize(src=input, dsize=(w_scale, h_scale), interpolation=cv2.INTER_CUBIC)
            label_scale = cv2.resize(src=label, dsize=(w_scale, h_scale), interpolation=cv2.INTER_CUBIC)
            label_scale[label_scale <= 200] = 1
            label_scale[label_scale > 200] = 0
            # plt.subplot(121), plt.imshow(input_scale,cmap='gray'), plt.title('input')
            # plt.subplot(122), plt.imshow(label_scale,cmap='gray'), plt.title('output')
            # plt.show()

            for k, v in transform.items():
                for i in range(0, h_scale - pitch_height - 1, stride):
                    for j in range(0, w_scale - pitch_width - 1, stride):
                        x.append(v(input_scale[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
                        y.append(v(label_scale[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
    return np.array(x), np.array(y)


def get_train_data():
    return generate_data(train_data_path, train_label_data_path)


def get_test_data():
    x = []
    test_files = os.listdir(test_data_path)
    for file in test_files:
        input = cv2.imread(filename=os.path.join(test_data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        h, w = input.shape
        while w % 4 != 0:
            w += 1
        while h % 4 != 0:
            h += 1
        img = cv2.resize(input, (w, h))
        x.append(img/255.0)
    return x

if __name__ =='__main__':
    get_train_data()