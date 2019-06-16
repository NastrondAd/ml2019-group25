from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class DataLoad(object):
    def __init__(self,file_name):
        self.file_name=file_name;
        dataset_zip=np.load(self.file_name,encoding = 'latin1')
        self.imgs= dataset_zip['imgs']
        self.latents_values  = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        # [ 1,  3,  6, 40, 32, 32]
        # color, shape, scale, orientation, posX, posY
        self.n_samples = 1000 # self.latents_sizes[::-1].cumprod()[-1]
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],np.array([1,])))
        # [737280, 245760, 40960, 1024, 32, 1]

    @property
    def sample_size(self):
        return self.n_samples

    def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
        latents = [0, shape, scale, orientation, x, y]
        index = np.dot(latents, self.latents_bases).astype(int)
        return self.get_images([index])[0]

    def get_images(self, indices):
        images = []
        for index in indices:
            img = self.imgs[index]
            img = img.reshape(4096)
            images.append(img)
            return images

    def get_random_images(self, size):
        indices = [np.random.randint(self.n_samples) for i in range(size)]
        return self.get_images(indices)
