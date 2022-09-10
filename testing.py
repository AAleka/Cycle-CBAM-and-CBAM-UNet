import glob
import time

from skimage.transform import resize
import math
import os
import warnings
import logging
import matplotlib
from skimage import color

from instancenormalization import InstanceNormalization
from tensorflow.keras.models import load_model
import numpy as np
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from skimage.metrics import structural_similarity
from matplotlib import pyplot
import cv2

from UNet.testing import segment

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
img_size = 256
directory = "results"


def preprocess_data(input_data):
    X = input_data
    X = (X - 127.5) / 127.5
    return X


def PSNR_SSIM(orig_img, gen_img):
    orig_img = (orig_img + 1) / 2.
    gen_img = (gen_img + 1) / 2.

    gray_orig_img = color.rgb2gray(orig_img)
    gray_gen_img = color.rgb2gray(gen_img)

    mse = np.mean((gray_orig_img - gray_gen_img) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    ssim = structural_similarity(gray_orig_img, gray_gen_img, multichannel=False, data_range=1.0)

    return round(psnr, 3), round(ssim, 3)


def save_plot(step, orig_bad, gen_good, psnr, ssim, folder):
    images = vstack((orig_bad, gen_good))
    titles = ['Original bad', 'Generated good']
    images = (images + 1) / 2.0

    f = pyplot.figure(figsize=(4.5, 2))
    for i in range(len(images)):
        f.add_subplot(1, len(images), i + 1)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])

    pyplot.figtext(0.5, 0.02, f"PSNR=[%.3f] SSIM=[%.3f]" % (psnr, ssim), fontsize=10, ha="center")

    pyplot.savefig(f"{directory}/testing2/{folder}/test_image{step}.png")
    pyplot.close()


file = open(f'{directory}/test.csv', 'r')
paths = file.readlines()

cust = {'InstanceNormalization': InstanceNormalization}
model_ArttoNoArt = load_model(f'{directory}/models/g_model_ArttoNoArt60000.h5', cust)
all_psnr = []
all_ssim = []
start = time.time()
# os.mkdir(f'{directory}/testing2/{step}')
# os.mkdir(f'{directory}/testing2/{step}/results')
for i, path in enumerate(paths):
    if path.split("\n")[0] != '0':
        bad_image = load_img(path.split("\n")[0], target_size=(img_size, img_size))
        bad_image = img_to_array(bad_image)

        bad_image_input = np.array([bad_image])
        bad_image_input = (bad_image_input - 127.5) / 127.5

        NoArt_generated = model_ArttoNoArt.predict(bad_image_input)

        all_psnr.append(PSNR_SSIM(bad_image_input[0], NoArt_generated[0])[0])
        all_ssim.append(PSNR_SSIM(bad_image_input[0], NoArt_generated[0])[1])

        bad_image_input[0] = (bad_image_input[0] + 1) / 2.0
        NoArt_generated[0] = (NoArt_generated[0] + 1) / 2.0

        # image = segment(bad_image_input[0], NoArt_generated[0])

        # cv2.putText(image, f'PSNR={all_psnr[-1]}', (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        # cv2.putText(image, f'SSIM={all_ssim[-1]}', (300, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        cv2.imwrite(f"results/hq/hq_{i}.png", cv2.cvtColor(NoArt_generated[0] * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"results/lq/lq_{i}.png", cv2.cvtColor(bad_image_input[0] * 255, cv2.COLOR_BGR2RGB))

# fig = pyplot.figure(figsize=(10, 7))
# pyplot.hist(all_psnr, bins=np.arange(start=min(all_psnr), stop=max(all_psnr), step=(max(all_psnr)+min(all_psnr))/100))
# pyplot.title("Second dataset")
# pyplot.xlabel("PSNR values")
# pyplot.ylabel("Number of occurrences")
# pyplot.savefig(f"{directory}/testing2/{step}/PSNR.png")
# pyplot.close()
#
# fig = pyplot.figure(figsize=(10, 7))
# pyplot.hist(all_ssim, bins=np.arange(start=min(all_ssim), stop=max(all_ssim), step=(max(all_ssim)+min(all_ssim))/100))
# pyplot.title("Second dataset")
# pyplot.xlabel("SSIM values")
# pyplot.ylabel("Number of occurrences")
# pyplot.savefig(f"{directory}/testing2/{step}/SSIM.png")
# pyplot.close()
f = open("time.txt", "w")
f.write(f"time = {(time.time()-start)/100}")
    # print("Average PSNR value:", sum(all_psnr) / len(all_psnr))
    # print("Average SSIM value:", sum(all_ssim) / len(all_ssim))
    # f = open(f"{directory}/testing2/{step}/metrics.txt", "w")
    # f.write(f'Average PSNR = {sum(all_psnr) / len(all_psnr)}' + "\n")
    # f.write(f'Average SSIM = {sum(all_ssim) / len(all_ssim)}' + "\n")
    # f.close()
