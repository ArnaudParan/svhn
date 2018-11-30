import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import to_categorical

from .digit_struct  import DigitStruct

def process_image(img, left, right, top, bottom):
    factor = 0.15
    width = right - left
    height = bottom - top
    left = left - np.round(factor * width)
    right = right + np.round(factor * width)
    top = top - np.round(factor * height)
    bottom = bottom + np.round(factor * height)

    big_crop = img.crop((left, top, right, bottom)).resize((64, 64))
    c_left, c_top = np.random.randint(0, 10, (2,))
    return big_crop.crop((c_left, c_top, c_left + 54, c_top + 54))

def get_image(basepath, name, left, top, height, width):
    tot_left = np.min(left)
    tot_right = np.max(left + width)
    tot_top = np.min(top)
    tot_bottom = np.max(top + height)
    with Image.open(os.path.join(basepath, name)) as img:
        image = np.array(process_image(img, tot_left, tot_right, tot_top, tot_bottom), dtype="float64")
    image -= np.mean(image, axis=(0, 1))
    image /= np.sqrt(np.mean(image * image, axis=(0, 1)))
    return image

def label_to_cats(label):
    cats = np.zeros((62,))
    length = min(len(label), 6)
    cats[length] = 1.
    label = np.pad(label, (0, 5 - min(len(label), 5)), "constant", constant_values=10)[:5]
    cats[7:] = to_categorical(label, num_classes=11).reshape((55,))

    return cats

def data_generator(data_path, bs, spe):
    ds = DigitStruct(os.path.join(data_path, "digitStruct.mat"))
    X = np.zeros((bs, 54, 54, 3))
    Y = np.zeros((bs, 62))
    ids = list(range(len(ds)))
    while True:
        random.shuffle(ids)
        for i in range(spe):
            for j, k in enumerate(ids[i*bs:(i+1)*bs]):
                dsl = ds[k]
                X[j] = get_image(data_path, dsl["name"], dsl["left"], dsl["top"], dsl["height"], dsl["width"])
                Y[j] = label_to_cats(dsl["label"])
            yield X, Y

def create_data(data_path, output_path, gen_factor=5):
    ds = DigitStruct(os.path.join(data_path, "digitStruct.mat"))
    labels = np.zeros((gen_factor * len(ds), 62))
    names = []
    for i, dsl in enumerate(ds):
        labels[i*gen_factor:(i+1)*gen_factor] = label_to_cats(dsl["label"])
        filename = dsl["name"]
        file_bname = ".".join(filename.split(".")[:-1])
        file_ext = filename.split(".")[-1]
        left = np.min(dsl["left"])
        right = np.max(dsl["left"] + dsl["width"])
        top = np.min(dsl["top"])
        bottom = np.max(dsl["top"] + dsl["height"])
        with Image.open(os.path.join(data_path, filename)) as img:
            for j in range(gen_factor):
                curr_filename = f"{file_bname}.{j}.{file_ext}"
                names.append(curr_filename)
                process_image(img, left, right, top, bottom).save(os.path.join(output_path, curr_filename))
    np.save(os.path.join(output_path, "labels.npy"), labels)
    pd.Series(names).to_csv(os.path.join(output_path, "names.csv"), index=False)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = next(data_generator("data/svhn/train/", 1, 1))
    print(y[0])
    plt.imshow(x[0])
    plt.show()
