import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from keras.models import load_model
from scipy import misc
import glob
import os
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = ""
imgs = glob.glob("/media/wj/2000G/xs/Datasets/celeba-HQ/celeba-64/*.jpg")

save_path = 'sample&model/'
sample_path = 'small_samples'
ae_sample_path = 'ae_samples'
model_path = 'models'

img_dim = 64
z_dim = 64
latent_dim = 2
img_channel = 3
g_model = load_model(save_path + model_path + '/g_model_148000.h5')
e_model = load_model(save_path + model_path + '/e_model_148000.h5')


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
            # np.random.shuffle(self.imgs)
            for i, f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                # f = f.astype(np.float32) / 255 * 2 - 1
                # X.append(f)
                if len(X) == self.batch_size or i == len(self.imgs) - 1:
                    X = np.array(X)
                    yield X
                    X = []  # x_train = np.load('./MNIST_data_npy/x_train_ssr.npy')


def sample_ae(base=0, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(imgs[int((i * n + j) / 2) + base * 50])]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
            j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    return figure


# x_sample = [imread(imgs[1])]
# z_sample = e_model.predict(np.array(x_sample))
# x_sample = g_model.predict(z_sample)
# digit = x_sample[0]
# plt.figure(1)
# plt.axis('off')
# plt.imshow(digit)
# plt.show()
if not os.path.exists('temp'):
    os.mkdir('temp')
for i in range(100):
    figure = sample_ae(base=i, n=10)
    imageio.imwrite('temp/%d.png' % i, figure)

# figure = sample_ae(n=10)
# imageio.imwrite('temp/%d.png' % base, figure)
# plt.figure(1)
# plt.imshow(figure)
# plt.show()
# x_sample = imread(imgs[18 * 50 + 15])
# x_sample = imread(imgs[7 * 50 + 33])

# x_sample = imread(imgs[23 * 50 + 33])
# z_sample = e_model.predict(np.array([x_sample]))
# print(z_sample)
# print(z_sample.shape)
