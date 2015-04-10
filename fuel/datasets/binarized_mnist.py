# -*- coding: utf-8 -*-
import logging
import os

from fuel import config
from fuel.datasets import H5PYDataset

logger = logging.getLogger(__name__)


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

    .. [DBN] Ruslan Salakhutdinov and Iain Murray, *On the Quantitative
       Analysis of Deep Belief Networks*, Proceedings of the 25th
       international conference on Machine learning, 2008, pp. 872-879.

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    which_set : 'train' or 'valid' or 'test'
        Whether to load the training set (50,000 samples) or the validation
        set (10,000 samples) or the test set (10,000 samples).

    """
    filename = 'binarized_mnist.hdf5'

    def __init__(self, which_set, load_in_memory=True, **kwargs):
        super(BinarizedMNIST, self).__init__(
            path=self.data_path, which_set=which_set,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
