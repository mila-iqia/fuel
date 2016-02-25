# -*- coding: utf-8 -*-
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class MNIST(H5PYDataset):
    u"""MNIST dataset.

    MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
    a database of handwritten digits. It is one of the most famous
    datasets in machine learning and consists of 60,000 training images
    and 10,000 testing images. The images are grayscale and 28 x 28 pixels
    large. It is accessible through Yann LeCun's website [LECUN].

    .. [LECUN] http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (60,000 examples) and the test
        set (10,000 examples).

    """
    filename = 'mnist.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(MNIST, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
