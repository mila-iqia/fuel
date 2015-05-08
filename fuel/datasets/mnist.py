# -*- coding: utf-8 -*-
import os

from fuel import config
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX


class MNIST(H5PYDataset):
    u"""MNIST dataset.

    MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
    a database of handwritten digits. It is one of the most famous
    datasets in machine learning and consists of 60,000 training images
    and 10,000 testing images. The images are grayscale and 28 x 28 pixels
    large. It is accessible through Yann LeCun's website [LECUN].

    .. [LBBH] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner,
       *Gradient-based learning applied to document recognition*,
       Proceedings of the IEEE, November 1998, 86(11):2278-2324.

    .. [LECUN] http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    which_set : 'train' or 'test'
        Whether to load the training set (60,000 samples) or the test set
        (10,000 samples).

    """
    filename = 'mnist.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(MNIST, self).__init__(self.data_path, which_set, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
