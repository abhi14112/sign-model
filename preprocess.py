import os
import cv2 as cv
import numpy as np
import random
path = "./data"
def preprocess(dataset_path, img_size=(98,98)):
    x = []
    y = []
    labels = os.listdir(dataset_path)
    labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
    print(labels)
    label_map = {label: idx for idx, label in enumerate(labels)}
    for label in labels:
        label_path = os.path.join(dataset_path, label)  
        print(label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)  
            img_grid = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            blur = cv.GaussianBlur(img_grid,(5,5),1.5)
            th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,9,2)
            ret, res = cv.threshold(th3, 80, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            img_grid = res
            img_grid = cv.resize(img_grid, img_size)
            img_grid = np.array(img_grid) / 255.0
            x.append(img_grid)
            y.append(label_map[label])
            x.append(img_grid)
            y.append(label_map[label])
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)
    return x, y
x, y = preprocess(path)
np.savez('preprocessed_data.npz', X=x, Y=y)