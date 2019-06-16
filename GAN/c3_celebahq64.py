import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from keras.models import load_model
from PIL import Image
from scipy import misc
import glob
import os

# x_train = np.load('./MNIST_data_npy/x_train_ssr.npy')
imgs = glob.glob("/media/wj/2000G/xs/Datasets/celeba-HQ/celeba-64/*.jpg")

os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
                    X = []


save_path = 'sample&model/'
sample_path = 'small_samples'
ae_sample_path = 'ae_samples'
model_path = 'models'

img_dim = 64
z_dim = 128
latent_dim = 4
img_channel = 3
g_model = load_model(save_path + model_path + '/g_model_148000.h5')
e_model = load_model(save_path + model_path + '/e_model_148000.h5')


# batch_size = 3
# img_data = img_generator(imgs, 'gan', batch_size).__iter__().__next__()


# x_sample = [imread(np.random.choice(imgs))]
# z_sample = e_model.predict(np.array(x_sample))
# x_sample = g_model.predict(z_sample)

# zc_sec = e_model.predict(img_data)


def gindex(index, min, max):
    x_sample = [imread(imgs[index])]
    zindex = e_model.predict(np.array(x_sample))
    c1_lin = np.linspace(min, max, 10)
    zimgs = np.zeros((10, z_dim + latent_dim))
    for i in range(10):
        zimgs[i] = zindex
        zimgs[i][z_dim + 3] = c1_lin[i]
        # zimgs[i][img_dim + 1] = c1_lin[i]
        # zimgs[i][img_dim] = c1_lin[i]
    outimgs = g_model.predict(zimgs)
    outimgs = (outimgs + 1) / 2 * 255
    outimgs = np.round(outimgs, 0).astype(int)
    # outimgs = outimgs[:, :, :, 0]
    return outimgs


# outimgs = gindex(6, -5, 5)
# print(len(outimgs))
# print(outimgs[0].shape)
# plt.figure()
# plt.imshow(outimgs[0])
# plt.show()

# print(outimgs.shape)
# zc_sec = e_model.predict(x_sec)
# # zc_sec[0][29] = 1
# # print(zc_sec.shape)
# z0 = zc_sec[46]
# # print(z0.shape)
# # print(z0)
# # g_sec = g_model.predict(zc_sec)
# # g_sec = 0.5 * g_sec + 0.5
# # # print(g_sec.shape)
# # g0 = g_sec[0, :, :, 0]
# # # print(g0.shape)
# # plt.figure()
# # plt.imshow(g0)
# # plt.show()
# c1_lin = np.linspace(-6, 4, 10)
# zimgs = np.zeros((10, 30))
# for i in range(10):
#     zimgs[i] = z0
#     zimgs[i][28] = c1_lin[i]
#
# outimgs = g_model.predict(zimgs)
# outimgs = 0.5 * outimgs + 0.5
# outimgs = outimgs[:, :, :, 0]

allimgs = np.zeros((3, 10, img_dim, img_dim, img_channel))
# allimgs[0] = gindex(zc_sec, 0, 1, 10)
# allimgs[1] = gindex(zc_sec, 8, -5, 5)
small = -5
large = 5
allimgs[0] = gindex(0 * 50 + 0, -15, -5)
allimgs[1] = gindex(7 * 50 + 33, small, large)
allimgs[2] = gindex(0 * 50 + 17, -13, -3)

allimgs = np.round(allimgs, 0).astype(int)
# image1 = allimgs[0][1]
# print(np.max(np.max(np.max(np.max(np.max(allimgs))))))
# print(np.min(np.min(np.min(np.min(np.min(allimgs))))))
#
# plt.figure()
# plt.imshow(allimgs[0][0])
# plt.show()

jizhongimg = np.zeros((3 * img_dim, 10 * img_dim, 3))

for k in range(3):
    # plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(10):
        jizhongimg[k * img_dim:(k + 1) * img_dim, i * img_dim:(i + 1) * img_dim, :] = allimgs[k][i]

jizhongimg = np.round(jizhongimg, 0).astype(int)
plt.figure(3)
plt.axis('off')
plt.imshow(jizhongimg)
plt.show()

# misc.imsave('c1_mnist_small.png', jizhongimg)
# im = Image.fromarray(jizhongimg)
# im.save("c1.jpg")
