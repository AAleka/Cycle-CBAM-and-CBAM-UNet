from random import random
import numpy as np
import math
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Reshape, multiply, Permute
from tensorflow.keras.layers import Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Dense, Add, Lambda
from tensorflow.keras import backend as K
from matplotlib import pyplot
from skimage.metrics import structural_similarity
import time
from skimage import color
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from instancenormalization import InstanceNormalization


def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)

    model = Model(in_image, patch_out)

    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Concatenate()([g, input_layer])

    return g


def cbam_block(input_feature, ratio=8):
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def res_cbam_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)

    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = cbam_block(g)
    g = Concatenate()([g, input_layer])

    return g


def channel_attention(input_feature, ratio):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(
        channel // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros'
    )
    shared_layer_two = Dense(
        channel,
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros'
    )

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def define_generator(image_shape, n_resnet=9):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)

    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    for _ in range(n_resnet):
        # g = resnet_block(256, g)
        g = res_cbam_block(256, g)
        # g = cbam_block(g)

    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True

    d_model.trainable = False  # False
    g_model_2.trainable = False  # False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    output_f = g_model_2(gen1_out)

    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                  loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model


def load_real_samples(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y


def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def save_models(g_model_AtoB, iteration):
    filename1 = f'retinal_model/models/g_model_ArttoNoArt{iteration}.h5'
    g_model_AtoB.save(filename1)

    print('>Saved: %s' % filename1)


def summarize_performance(step, g_model, trainX, name, n_samples=5):
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)

    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0

    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])

    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])

    filename1 = 'retinal_model/training/%s_generated_plot_%06d.png' % (name, step)
    pyplot.savefig(filename1)
    pyplot.close()


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


def save_generated(step, g_model, trainX, n_samples=1):
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)

    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0

    pyplot.imshow(X_in[0])
    pyplot.savefig(f"retinal_model/Original bad/{step}_real.png")

    pyplot.imshow(X_out[0])
    pyplot.savefig(f"retinal_model/Original bad/{step}_fake.png")


def PSNR_SSIM(g_model, trainX):
    X_in, _ = generate_real_samples(trainX, 1, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)

    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0

    gray_X_in = color.rgb2gray(X_in[0])
    gray_X_out = color.rgb2gray(X_out[0])

    mse = np.mean((gray_X_in - gray_X_out) ** 2)

    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    ssim = structural_similarity(gray_X_in, gray_X_out, multichannel=False, data_range=1.0)

    return psnr, ssim


def validate(g_model_AtoB):
    file = open('retinal_model/validation.csv', 'r')
    paths = file.readlines()

    psnr = 0
    ssim = 0

    for i, path in enumerate(paths):
        if path.split("\n")[0] != '0':
            bad_image = load_img(path.split("\n")[0], target_size=(256, 256))
            bad_image = img_to_array(bad_image)

            bad_image_input = np.array([bad_image])
            bad_image_input = (bad_image_input - 127.5) / 127.5

            NoArt_generated = g_model_AtoB.predict(bad_image_input)

            bad_image_input = (bad_image_input + 1) / 2.
            NoArt_generated = (NoArt_generated + 1) / 2.

            gray_orig_img = color.rgb2gray(bad_image_input[0])
            gray_gen_img = color.rgb2gray(NoArt_generated[0])

            mse = np.mean((gray_orig_img - gray_gen_img) ** 2)
            if mse == 0:
                psnr += 100
            else:
                max_pixel = 1.0
                psnr += 20 * math.log10(max_pixel / math.sqrt(mse))

            ssim += structural_similarity(gray_orig_img, gray_gen_img, multichannel=False, data_range=1.0)

    return psnr / len(paths), ssim / len(paths)


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, n_epochs, n_iters):
    n_batch = 1
    n_patch = d_model_A.output_shape[1]

    poolA, poolB = list(), list()

    bat_per_epo = int(len(dataset[0]) / n_batch)

    n_steps = bat_per_epo

    start = time.time()

    psnr = 0
    ssim = 0
    all_psnr = []
    all_ssim = []

    all_psnr_valid = []
    all_ssim_valid = []
    checkpoint = 200
    count = 0
    for epoch in range(n_epochs):
        for iter in range(n_iters):
            trainA = dataset[epoch]
            trainB = dataset[epoch + n_epochs]
            for i in range(n_steps):
                count += 1
                X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
                X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

                X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
                X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

                X_fakeA = update_image_pool(poolA, X_fakeA)
                X_fakeB = update_image_pool(poolB, X_fakeB)

                g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB,
                                                                                       X_realA])

                dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
                dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

                g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA,
                                                                                       X_realB])

                dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
                dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

                psnr = psnr + PSNR_SSIM(g_model_AtoB, trainA)[0]
                ssim = ssim + PSNR_SSIM(g_model_AtoB, trainA)[1]

                if count % (2 * checkpoint) == 0 and count > 20000:
                    save_generated(count, g_model_AtoB, trainA)

                if count % checkpoint == 0:
                    summarize_performance(count, g_model_AtoB, trainA, 'AtoB')

                    all_psnr.append(psnr / checkpoint)
                    all_ssim.append(ssim / checkpoint)
                    print('Iteration>%d, g[%.3f] PSNR[%.3f dB] SSIM[%.3f] time[%.3f min]' % (
                              count, g_loss1, all_psnr[-1], all_ssim[-1], (time.time() - start) / 60))

                    psnr = 0
                    ssim = 0
                    start = time.time()

                if count % (5 * checkpoint) == 0:
                    all_psnr_valid.append(validate(g_model_AtoB)[0])
                    all_ssim_valid.append(validate(g_model_AtoB)[1])
                    print('Validation PSNR[%.3f dB] Validation SSIM[%.3f]' % (all_psnr_valid[-1], all_ssim_valid[-1]))

                if count % 2000 == 0 and count > 20000:
                    save_models(g_model_AtoB, count)

    plot_metrics(all_psnr, all_ssim)
    plot_metrics(all_psnr_valid, all_ssim_valid, "validation")


def plot_metrics(all_psnr, all_ssim, flag="training"):
    if flag == "training":
        pyplot.plot(np.arange(0, len(all_psnr), 1, dtype=int), all_psnr)
        pyplot.ylabel("Average PSNR value")
        pyplot.xlabel("Steps")
        pyplot.savefig(f"retinal_model/training_PSNR.png")
        pyplot.close()

        pyplot.plot(np.arange(0, len(all_ssim), 1, dtype=int), all_ssim)
        pyplot.ylabel("Average SSIM value")
        pyplot.xlabel("Steps")
        pyplot.savefig(f"retinal_model/training_SSIM.png")
        pyplot.close()
    elif flag == "validation":
        pyplot.plot(np.arange(0, len(all_psnr), 1, dtype=int), all_psnr)
        pyplot.ylabel("Average PSNR value")
        pyplot.xlabel("Steps")
        pyplot.savefig(f"retinal_model/validation_PSNR.png")
        pyplot.close()

        pyplot.plot(np.arange(0, len(all_ssim), 1, dtype=int), all_ssim)
        pyplot.ylabel("Average SSIM value")
        pyplot.xlabel("Steps")
        pyplot.savefig(f"retinal_model/validation_SSIM.png")
        pyplot.close()
