# -*- coding: utf-8 -*-
from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path


class BinarizedMNIST(H5PYDataset):
    u"""Binarized, unlabeled MNIST dataset.

    MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
    a database of handwritten digits. It is one of the most famous datasets
    in machine learning and consists of 60,000 training images and 10,000
    testing images. The images are grayscale and 28 x 28 pixels large.

    This particular version of the dataset is the one used in R.
    Salakhutdinov's DBN paper [DBN] as well as the VAE and NADE papers, and
    is accessible through Hugo Larochelle's public website [HUGO].

    The training set has further been split into a training and a
    validation set. All examples were binarized by sampling from a binomial
    distribution defined by the pixel values.

    .. [LBBH] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner,
       *Gradient-based learning applied to document recognition*,
       Proceedings of the IEEE, November 1998, 86(11):2278-2324.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'valid' and 'test',
        corresponding to the training set (50,000 examples), the validation
        set (10,000 samples) and the test set (10,000 examples).

    """
    filename = 'binarized_mnist.hdf5'

    def __init__(self, which_sets, load_in_memory=True, **kwargs):
        super(BinarizedMNIST, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets,
            load_in_memory=load_in_memory, **kwargs)
