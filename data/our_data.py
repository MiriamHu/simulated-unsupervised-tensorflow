import numpy as np
import os
from glob import glob

from layers import image_from_paths
from utils import imread
import cv2

__author__ = 'mhuijser'


def create_tree(config, data_path, rng):
    if not os.path.exists(data_path):
        print('creating folder', data_path)
        os.makedirs(data_path)

    synthetic_image_path = os.path.join(data_path, config.synthetic_image_dir)

    return synthetic_image_path


class DataLoader(object):
    def __init__(self, config, rng=None):
        self.rng = np.random.RandomState(1) if rng is None else rng

        self.data_path = os.path.join(config.data_dir, 'our')
        self.real_data_path = os.path.join(self.data_path, config.real_image_dir)
        self.batch_size = config.batch_size
        self.debug = config.debug

        synthetic_image_path = create_tree(config, self.data_path, rng)

        self.synthetic_data_paths = np.array(glob(os.path.join(synthetic_image_path, '*.jpg')))
        self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0]).shape[:2]) + [1]

        self.real_data_paths = np.array(glob(os.path.join(self.real_data_path, "*.jpg")))
        self.real_data_dims = list(imread(self.real_data_paths[0]).shape[:2]) + [1]

        self.synthetic_data_paths.sort()

        # if np.rank(self.real_data) == 3:
        #     self.real_data = np.expand_dims(self.real_data, -1)

        self.real_p = 0

    # def get_observation_size(self):
    #     return self.real_data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.real_p = 0

    def __iter__(self):
        return self

    # def __next__(self, n=None):
    #     """ n is the number of examples to fetch """
    #     if n is None: n = self.batch_size
    #
    #     if self.real_p == 0:
    #         inds = self.rng.permutation(self.real_data.shape[0])
    #         self.real_data = self.real_data[inds]
    #
    #     if self.real_p + n > self.real_data.shape[0]:
    #         self.reset()
    #
    #     x = self.real_data[self.real_p: self.real_p + n]
    #     self.real_p += self.batch_size
    #
    #     return x

    def __next__(self, n=None):
        """

        :param n: the number of examples to fetch
        :return:
        """
        if n is None:
            n = self.batch_size
        if self.real_p == 0:
            inds = self.rng.permutation(len(self.real_data_paths))
            self.real_data_paths = self.real_data_paths[inds]

        if self.real_p + n > len(self.real_data_paths):
            self.reset()

        paths = self.real_data_paths[self.real_p: self.real_p + n]
        self.real_p += self.batch_size

        # real_filenames, real_data = image_from_paths(paths, self.real_data_dims)
        real_data = np.expand_dims(np.stack(
            [cv2.cvtColor( imread(path), cv2.COLOR_BGR2GRAY) for path in paths]
        ), -1)
        return real_data

    next = __next__
