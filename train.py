import logging
import pandas as pd
import os
import gc
import random
import time
import warnings
import matplotlib
from os import listdir
import numpy as np
from numpy import asarray
from numpy.random import randint
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from model import define_generator, define_discriminator, define_composite_model, train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
img_size = 256


def load_images(img_path1, img_path2="", size=(img_size, img_size), flag='low'):
    data_list = list()
    if flag == 'low':
        valid_path = list()
        test_path = list()
        for filename in listdir(img_path1):
            if np.random.rand() < 0.85:
                pixels = load_img(img_path1 + filename, target_size=size)
                pixels = img_to_array(pixels)
                data_list.append(pixels)
            else:
                if np.random.rand() < 0.85:
                    test_path.append(img_path1 + filename)
                else:
                    valid_path.append(img_path1 + filename)
        if img_path2 != "":
            for filename in listdir(img_path2):
                if np.random.rand() < 0.85:
                    pixels = load_img(img_path2 + filename, target_size=size)
                    pixels = img_to_array(pixels)
                    data_list.append(pixels)
                else:
                    if np.random.rand() < 0.85:
                        test_path.append(img_path2 + filename)
                    else:
                        valid_path.append(img_path2 + filename)

        return asarray(data_list), valid_path, test_path

    else:
        for filename in listdir(img_path1):
            pixels = load_img(img_path1 + filename, target_size=size)
            pixels = img_to_array(pixels)
            data_list.append(pixels)
        if img_path2 != "":
            for filename in listdir(img_path2):
                pixels = load_img(img_path2 + filename, target_size=size)
                pixels = img_to_array(pixels)
                data_list.append(pixels)

        return asarray(data_list)


def preprocess_data(input_data):
    X = input_data
    X = (X - 127.5) / 127.5
    return X


n_epochs = 6
n_iters = 10
n_LQ = 1000
n_HQ = 1000

os.mkdir(f'results')
os.mkdir(f'results/models')
os.mkdir(f'results/Original bad')
os.mkdir(f'results/testing')
os.mkdir(f'results/training')

start = time.time()

path1 = 'datasets/EyeQ/'
# path2 = 'datasets/artifacts_no_artifacts/'

dataLQ_train, valid_path, test_path = load_images(path1 + '1/')  # , path2 + 'artifacts/'
print('Loaded training low-quality images: ', dataLQ_train.shape)
print('Number of validation low-quality images: ', len(valid_path))
print('Number of testing low-quality images: ', len(test_path))

random.shuffle(test_path)
test_df = pd.DataFrame(test_path)
test_df.to_csv('results/test.csv', index=False)

random.shuffle(valid_path)
valid_df = pd.DataFrame(valid_path)
valid_df.to_csv('results/validation.csv', index=False)

del test_path, valid_path, test_df, valid_df
gc.collect()

dataHQ_all = load_images(path1 + '0/', flag='high')  # , path2 + 'no_artifacts/'
print('Loaded training high-quality images: ', dataHQ_all.shape)

dataset = []

for i in range(n_epochs):
    dataLQ = resample(dataLQ_train, replace=False, n_samples=n_LQ)
    dataLQ = preprocess_data(dataLQ)
    dataset.append(dataLQ)
    del dataLQ
    gc.collect()

for i in range(n_epochs):
    dataHQ = resample(dataHQ_all, replace=False, n_samples=n_HQ)
    dataHQ = preprocess_data(dataHQ)
    dataset.append(dataHQ)
    del dataHQ
    gc.collect()

image_shape = dataset[0].shape[1:]
g_model_ArttoNoArt = define_generator(image_shape)
g_model_NoArttoArt = define_generator(image_shape)
d_model_Art = define_discriminator(image_shape)
d_model_NoArt = define_discriminator(image_shape)
c_model_ArttoNoArt = define_composite_model(g_model_ArttoNoArt, d_model_NoArt, g_model_NoArttoArt, image_shape)
c_model_NoArttoArt = define_composite_model(g_model_NoArttoArt, d_model_Art, g_model_ArttoNoArt, image_shape)

print("Preparation time:", (time.time()-start)/60, "mins")

print("Training started...")
start = time.time()

train(d_model_Art, d_model_NoArt, g_model_ArttoNoArt, g_model_NoArttoArt, c_model_ArttoNoArt,
      c_model_NoArttoArt, dataset, n_epochs, n_iters)

del dataLQ_train, dataHQ_all, dataset
gc.collect()

print("Training time:", ((time.time()-start)/60) / 60, "hours")
