import os

import cv2
import numpy as np

import random

train_data_x_path = "./data/val_blur"
train_data_y_path = "./data/val_sharp"
test_data_path = "./data/test_blur"
save_path = "./data/npy"
y_npy = "y_patches.npy"
x_npy = "x_patches.npy"
y_val_npy = "y_val_patches.npy"
x_val_npy = "x_val_patches.npy"
pitch_height = 128
pitch_width = 128
stride = 128
scales = (1,)
transform = {
    "origin": lambda img: img,
    # "rot90_1":lambda img:np.rot90(img),
    # "rot90_2":lambda img:np.rot90(img,2),
    # "rot90_3":lambda img:np.rot90(img,3),
    # "flip":lambda img:np.flipud(img),
    # "flip_rot90_1":lambda img:np.flipud(np.rot90(img)),
    # "flip_rot90_2": lambda img: np.flipud(np.rot90(img,2)),
    # "flip_rot90_3": lambda img: np.flipud(np.rot90(img,3)),
}


def generate_data(files):
    x = []
    for file in files:
        img = cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        h, w = img.shape[0:2]
        for scale in scales:
            # 切片左闭右开
            h_scale, w_scale = int(h * scale), int(w * scale)
            img_scale = cv2.resize(src=img, dsize=(w_scale, h_scale), interpolation=cv2.IMREAD_GRAYSCALE)
            for k, v in transform.items():
                for i in range(0, h_scale - pitch_height - 1, stride):
                    for j in range(0, w_scale - pitch_width - 1, stride):
                        x.append(
                            img_scale(v(img[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis]))

    return np.array(x)


def generate_test_data(data_path):
    files = os.listdir(data_path)
    x = []
    y = []
    for file in files:
        img = cv2.imread(filename=os.path.join(data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        w, h = img.shape
        while w % 32:
            w += 1
        while h % 32:
            h += 1
        img = cv2.resize(src=img, dsize=(w, h), interpolation=cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename=os.path.join(data_path,file),flags=cv2.IMREAD_COLOR)
        img = img / 255.0
        x.append(img)
    return y, x


def get_train_data():
    if not os.path.exists(os.path.join(save_path, y_npy)) or not os.path.exists(
            os.path.join(save_path, x_npy)):
        save_as_npy()
    return np.load(os.path.join(save_path, x_npy)), np.load(os.path.join(save_path, y_npy))


def get_validation_data():
    if not os.path.exists(os.path.join(save_path, y_val_npy)) or not os.path.exists(
            os.path.join(save_path, x_val_npy)):
        save_as_npy()
    return np.load(os.path.join(save_path, x_val_npy)), np.load(os.path.join(save_path, y_val_npy))


def get_test_data():
    return generate_test_data(test_data_path)


def save_as_npy():
    files_x = getFiles(train_data_x_path)
    files_y = getFiles(train_data_y_path)
    train_size = int(len(files_x) * 0.95)
    x = generate_data(files_x[0:train_size])
    y = generate_data(files_y[0:train_size])
    x_val = generate_data(files_x[train_size:len(files_x)])
    y_val = generate_data(files_y[train_size:len(files_y)])
    print('Shape of result = ' + str(y.shape))
    print('Saving data...')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, x_npy), x)
    np.save(os.path.join(save_path, y_npy), y)
    np.save(os.path.join(save_path, x_val_npy), x_val)
    np.save(os.path.join(save_path, y_val_npy), y_val)
    print('Done.')


def getFiles(path):
    files = []
    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            files.append(os.path.join(os.path.join(path, dir), file))
    return files


def loadData(file_num):
    files_x = getFiles(train_data_x_path)
    files_y = getFiles(train_data_y_path)
    train_size = int(len(files_x) * 0.95)
    input_x = []
    input_y = []
    num = 1

    idx = list(range(train_size))
    random.shuffle(idx)
    for i in idx:
        img_x = cv2.imread(files_x[i], flags=cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(files_y[i], flags=cv2.IMREAD_GRAYSCALE)
        input_x.extend(getImage(img_x))
        input_y.extend(getImage(img_y))
        num += 1
        if num % file_num == 0:
            yield np.asarray(input_x), np.asarray(input_y), train_size / file_num
            input_x = []
            input_y = []


def getImage(img):
    img = img / 255.0
    h, w = img.shape[0:2]
    x = []
    for scale in scales:
        # 切片左闭右开
        h_scale, w_scale = int(h * scale), int(w * scale)
        img_scale = cv2.resize(src=img, dsize=(w_scale, h_scale), interpolation=cv2.IMREAD_GRAYSCALE)
        for k, v in transform.items():
            for i in range(0, h_scale - pitch_height - 1, stride):
                for j in range(0, w_scale - pitch_width - 1, stride):
                    x.append(
                        v(img_scale[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
    return x


if __name__ == '__main__':
    save_as_npy()
