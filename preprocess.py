import os
import cv2 as cv
import numpy as np
import random
path = "./data"
def preprocess(dataset_path, img_size=(100,100)):
    x = []
    y = []
    labels = os.listdir(dataset_path)
    label_map = {label: idx for idx, label in enumerate(labels)}
    for label in labels:
        label_path = os.path.join(dataset_path, label)  # Fixed the variable name here
        print(label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)  # Fixed the variable name here
            img_grid = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            image = np.array(img_grid) / 255.0
            new_array = cv.resize(image, img_size)
            x.append(new_array)
            y.append(label_map[label])
            x.append(new_array)
            y.append(label_map[label])
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)
    return x, y
x, y = preprocess(path)
np.savez('preprocessed_data.npz', X=x, Y=y)