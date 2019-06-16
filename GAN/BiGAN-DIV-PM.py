#! -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
import os
from keras.callbacks import ModelCheckpoint

save_path = 'sample&model/'
sample_path = 'small_samples'
ae_sample_path = 'ae_samples'
model_path = 'models'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(save_path + sample_path):
    os.mkdir(save_path + sample_path)

if not os.path.exists(save_path + ae_sample_path):
    os.mkdir(save_path + ae_sample_path)

if not os.path.exists(save_path + model_path):
    os.mkdir(save_path + model_path)

# imgs = glob.glob('../../CelebA-HQ/train/*.png')
# np.random.shuffle(imgs)
# img_dim = 128
# z_dim = 128
# num_layers = int(np.log2(img_dim)) - 3
# max_num_channels = img_dim * 8
# f_size = img_dim // 2 ** (num_layers + 1)
# batch_size = 64
# imgs = np.load('/media/wj/hdd/xs/Datasets/MNIST_SR/x_train_ssr.npy')
imgs = glob.glob("/media/wj/2000G/xs/Datasets/celeba-HQ/celeba-64/*.jpg")
np.random.shuffle(imgs)
img_dim = 64
z_dim = 128
img_channel = 3
batch_size = 64
latent_dim = 4
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2 ** (num_layers)
ze_dim = 64


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        return x.astype(np.float32) / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """

    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i, f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                # f = f.astype(np.float32) / 255 * 2 - 1
                # X.append(f)
                if len(X) == self.batch_size or i == len(self.imgs) - 1:
                    X = np.array(X)
                    yield X
                    X = []


# 编码器1（为了编码）
x_in = Input(shape=(img_dim, img_dim, img_channel))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2 ** (num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(z_dim + latent_dim)(x)

e_model = Model(x_in, x)
e_model.summary()

# 编码器2（为了判别器）
x_in = Input(shape=(img_dim, img_dim, img_channel))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2 ** (num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(ze_dim, use_bias=False)(x)

te_model = Model(x_in, x)
te_model.summary()

# 判别器
z_in = Input(shape=(ze_dim + z_dim + latent_dim,))  # 2倍zdim，一半是z，一半是经过te_model的x
z = z_in

z = Dense(1024, use_bias=False)(z)
z = LeakyReLU(0.2)(z)
z = Dense(1024, use_bias=False)(z)
z = LeakyReLU(0.2)(z)
# z = Dense(7 ** 2 * max_num_channels)(z)
# z = BatchNormalization()(z)
# z = Activation('relu')(z)
# z = Reshape((7, 7, max_num_channels))(z)
# for i in range(num_layers + 1):
#     num_channels = max_num_channels // 2 ** (num_layers - i)
#     z = Conv2D(num_channels,
#                (5, 5),
#                strides=(2, 2),
#                use_bias=False,
#                padding='same')(z)
#     if i > 0:
#         z = BatchNormalization()(z)
#     z = LeakyReLU(0.2)(z)
# z = Flatten()(z)
# # x = Dense(z_dim, use_bias=False)(x)
z_d = Dense(1, use_bias=False)(z)

td_model = Model(z_in, z_d)
td_model.summary()

# 生成器

max_num_channels = img_dim * 8
f_size = img_dim // 2 ** (num_layers + 1)
z_in = Input(shape=(z_dim + latent_dim,))
z = z_in

z = Dense(f_size ** 2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2 ** (i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(img_channel,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()

# 2个对c进行解耦的网络

# Dis1
zc_in = Input(shape=(z_dim + latent_dim - 1,))
zc = zc_in

zc = Dense(512, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
zc = Dense(1024, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
# x = Dense(z_dim, use_bias=False)(x)
zc = Dense(1, use_bias=True)(zc)

distanglec1_model = Model(zc_in, zc)
distanglec1_model.summary()

# Dis2
zc_in = Input(shape=(z_dim + latent_dim - 1,))
zc = zc_in

zc = Dense(512, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
zc = Dense(1024, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
# x = Dense(z_dim, use_bias=False)(x)
zc = Dense(1, use_bias=True)(zc)

distanglec2_model = Model(zc_in, zc)
distanglec2_model.summary()

# Dis3
zc_in = Input(shape=(z_dim + latent_dim - 1,))
zc = zc_in

zc = Dense(512, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
zc = Dense(1024, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
# x = Dense(z_dim, use_bias=False)(x)
zc = Dense(1, use_bias=True)(zc)

distanglec3_model = Model(zc_in, zc)
distanglec3_model.summary()

# Dis4
zc_in = Input(shape=(z_dim + latent_dim - 1,))
zc = zc_in

zc = Dense(512, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
zc = Dense(1024, use_bias=False)(zc)
zc = LeakyReLU(0.2)(zc)
# x = Dense(z_dim, use_bias=False)(x)
zc = Dense(1, use_bias=True)(zc)

distanglec4_model = Model(zc_in, zc)
distanglec4_model.summary()

k = 2
p = 6


def slice(x, start=None, end=None):
    return x[:, start:end]


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, img_channel))
z_in = Input(shape=(z_dim,))
c_in = Input(shape=(latent_dim,))

g_model.trainable = False
e_model.trainable = False
distanglec1_model.trainable = False
distanglec2_model.trainable = False

x_real, z_fake = x_in, z_in
c_fake = c_in
zc_fake = Concatenate()([z_fake, c_fake])
x_fake = g_model(zc_fake)
zc_real = e_model(x_real)

z_real = Lambda(slice, arguments={'end': z_dim})(zc_real)
c_real = Lambda(slice, arguments={'start': z_dim})(zc_real)

x_real_encoded = te_model(x_real)
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded, z_real, c_real])
xz_fake = Concatenate()([x_fake_encoded, z_fake, c_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)

d_train_model = Model([x_in, z_in, c_in],
                      [xz_real_score, xz_fake_score])

d_loss = K.mean(xz_real_score - xz_fake_score)
# d_loss = d_loss[:, 0]
real_grad = K.gradients(xz_real_score, [x_real])[0]
fake_grad = K.gradients(xz_fake_score, [x_fake])[0]

real_grad_norm = K.sum(real_grad ** 2, axis=[1, 2, 3]) ** (p / 2)

fake_grad_norm = K.sum(fake_grad ** 2, axis=[1, 2, 3]) ** (p / 2)
grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2
# g_model
# + K.mean(K.square(c_real - info_c_real))

w_dist = K.mean(xz_fake_score - xz_real_score)

d_train_model.add_loss(d_loss + grad_loss)

d_train_model_filepath = "./small_samples/d_train_model_best.hdf5"
d_train_model_checkpoint = ModelCheckpoint(d_train_model_filepath, monitor='d_loss + grad_loss',
                                           save_best_only=True)
d_train_model_checkpoint_ = [d_train_model_checkpoint]

d_train_model.compile(optimizer=Adam(2e-4, 0.5))
d_train_model.metrics_names.append('w_dist')
d_train_model.metrics_tensors.append(w_dist)

# d_loss = xz_real_score - xz_fake_score
# d_loss = d_loss[:, 0]
# d_norm = 10 * (K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3]) + K.mean(K.abs(z_real - z_fake), axis=1))
# d_loss = K.mean(- d_loss + 0.5 * d_loss ** 2 / d_norm)
#
# d_train_model.add_loss(d_loss)
# d_train_model.compile(optimizer=Adam(2e-4, 0.5))

# 整合模型（训练生成器）
g_model.trainable = True
e_model.trainable = True
distanglec1_model.trainable = True
distanglec2_model.trainable = True
distanglec3_model.trainable = True
distanglec4_model.trainable = True
td_model.trainable = False
te_model.trainable = False

x_real, z_fake = x_in, z_in
c_fake = c_in

zc_fake = Concatenate()([z_fake, c_fake])
x_fake = g_model(zc_fake)
zc_real = e_model(x_real)

z_real = Lambda(slice, arguments={'end': z_dim})(zc_real)
c_real = Lambda(slice, arguments={'start': z_dim})(zc_real)
z_real_ = Lambda(lambda x: K.stop_gradient(x))(z_real)
zc_real_ = Lambda(lambda x: K.stop_gradient(x))(zc_real)
x_real_ = g_model(zc_real_)
x_fake_ = Lambda(lambda x: K.stop_gradient(x))(x_fake)
zc_fake_ = e_model(x_fake_)
z_fake_ = Lambda(slice, arguments={'end': z_dim})(zc_fake_)
c_fake_ = Lambda(slice, arguments={'start': z_dim})(zc_fake_)

x_real_encoded = te_model(x_real)
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded, z_real, c_real])
xz_fake = Concatenate()([x_fake_encoded, z_fake, c_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)


def cfenli(c, i):
    ci = Lambda(slice, arguments={'start': i, 'end': i + 1})(c)
    c_front_i = Lambda(slice, arguments={'end': i})(c)
    c_back_i = Lambda(slice, arguments={'start': i + 1})(c)
    cnoi = Concatenate()([c_front_i, c_back_i])
    return ci, cnoi


[c0, cno0] = cfenli(c_fake, 0)
[c1, cno1] = cfenli(c_fake, 1)
[c2, cno2] = cfenli(c_fake, 2)
[c3, cno3] = cfenli(c_fake, 3)
zc_0 = Concatenate()([z_fake, cno0])
zc_1 = Concatenate()([z_fake, cno1])
zc_2 = Concatenate()([z_fake, cno2])
zc_3 = Concatenate()([z_fake, cno3])

dis_c0 = distanglec1_model(zc_0)
dis_c1 = distanglec1_model(zc_1)
dis_c2 = distanglec1_model(zc_2)
dis_c3 = distanglec1_model(zc_3)

# c1 = Lambda(slice, arguments={'end': 1})(c_fake)
# c2 = Lambda(slice, arguments={'start': 1})(c_fake)
# zc_1 = Concatenate()([z_fake, c1])
# dis_c2 = distanglec2_model(zc_1)
# zc_2 = Concatenate()([z_fake, c2])
# dis_c1 = distanglec1_model(zc_2)

g_train_model = Model([x_in, z_in, c_in],
                      [xz_real_score, xz_fake_score])

distangle_loss = K.mean(K.square(c0 - dis_c0)) + K.mean(K.square(c1 - dis_c1)) + K.mean(K.square(c2 - dis_c2)) + K.mean(
    K.square(c3 - dis_c3))

g_loss = K.mean(xz_fake_score) \
         + 2 * K.mean(K.square(z_fake - z_fake_)) \
         + 0.05 * K.mean(K.square(x_real - x_real_)) \
         + 2 * K.mean(K.square(c_fake - c_fake_))

g_train_model.add_loss(g_loss - distangle_loss)
g_train_model_filepath = "./small_samples/g_train_model_best.hdf5"
g_train_model_checkpoint = ModelCheckpoint(g_train_model_filepath, monitor='g_loss',
                                           save_best_only=True)
g_train_model_checkpoint_ = [g_train_model_checkpoint]
g_train_model.compile(optimizer=Adam(2e-4, 0.5))

# g_loss = K.mean(xz_real_score - xz_fake_score) + 2 * K.mean(K.square(z_fake - z_fake_)) + 3 * K.mean(
#     K.square(x_real - x_real_))
#
# g_train_model.add_loss(g_loss)
# g_train_model.compile(optimizer=Adam(2e-4, 0.5))

g_train_model.metrics_names.append('d_loss')
g_train_model.metrics_tensors.append(K.mean(xz_real_score - xz_fake_score))
g_train_model.metrics_names.append('r_loss')
g_train_model.metrics_tensors.append(
    2 * K.mean(K.square(z_fake - z_fake_)) + 1 * K.mean(K.square(x_real - x_real_)) + 2 * K.mean(
        K.square(c_fake - c_fake_)))
g_train_model.metrics_names.append('distangle_loss')
g_train_model.metrics_tensors.append(-distangle_loss)

# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# print(2)


# 采样函数
def sample(path, n=8, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, img_channel))
    # if z_samples is None:
    #     z_samples = np.random.randn(n ** 2, z_dim)
    z_samples = np.random.randn(n ** 2, z_dim + latent_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
            j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    # figure = 0.5 * figure + 0.5
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
            j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


if __name__ == '__main__':

    iters_per_sample = 100
    iters_per_model = 2000
    total_iter = 150000
    n_size = 8
    img_data = img_generator(imgs, 'gan', batch_size).__iter__()
    Z = np.random.randn(n_size ** 2, z_dim)
    if os.path.exists(save_path + model_path + '/g_train_model_20000.weights'):
        g_train_model.load_weights(save_path + model_path + '/g_train_model_20000.weights')
    # print(3)
    for i in range(total_iter):
        for j in range(2):
            x_sample = img_data.__next__()
            z_sample = np.random.randn(len(x_sample), z_dim)
            c_sample = np.random.randn(len(x_sample), latent_dim)
            # print(4)
            d_loss = d_train_model.fit(
                [x_sample, z_sample, c_sample], None, callbacks=d_train_model_checkpoint_, verbose=0)
        # print(1)
        for j in range(1):
            x_sample = img_data.__next__()
            z_sample = np.random.randn(len(x_sample), z_dim)
            c_sample = np.random.randn(len(x_sample), latent_dim)
            # g_loss = g_train_model.train_on_batch(
            #     [x_sample, z_sample], None)
            g_loss = g_train_model.fit(
                [x_sample, z_sample, c_sample], None, callbacks=g_train_model_checkpoint_, verbose=0)

        if i % 10 == 0:
            print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss.history, g_loss.history))
        if i % iters_per_sample == 0:
            sample(save_path + sample_path + '/test_%s.png' % i, n_size, Z)
            sample_ae(save_path + ae_sample_path + '/test_ae_%s.png' % i)
        if i % iters_per_model == 0:
            g_train_model.save_weights(save_path + model_path + ('/g_train_model_%d.weights' % i))
            g_model.save(save_path + model_path + ('/g_model_%d.h5' % i))
            e_model.save(save_path + model_path + ('/e_model_%d.h5' % i))
