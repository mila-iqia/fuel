# -*- coding: utf-8 -*-
import logging
import os
from collections import OrderedDict

import numpy

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes

logger = logging.getLogger(__name__)


@do_not_pickle_attributes('indexables')
class BinarizedMNIST(IndexableDataset):
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
    provides_sources = ('features',)
    folder = 'binarized_mnist'
    files = {set_: 'binarized_mnist_{}.npy'.format(set_) for set_ in
             ('train', 'valid', 'test')}

    def __init__(self, which_set, flatten=True, **kwargs):
        if which_set not in ('train', 'valid', 'test'):
            raise ValueError("available splits are 'train', 'valid' and "
                             "'test'")
        self.which_set = which_set
        self.flatten = flatten

        super(BinarizedMNIST, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_binarized_mnist())), **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.folder,
                            self.files[self.which_set])

    def load(self):
        indexable, = self._load_binarized_mnist()
        self.indexables = [indexable[self.start:self.stop]]

    def _load_binarized_mnist(self):
        if os.path.isfile(self.data_path):
            images = numpy.load(self.data_path).astype(config.floatX)
        else:
            logger.warn("The faster .npy version of binarized_mnist_{} isn't "
                        "available, falling back to the .amat version."
                        .format(self.which_set))
            images = numpy.loadtxt(self.data_path[:-3] + 'amat',
                                   dtype=config.floatX)
        if not self.flatten:
            images = images.reshape((len(images), 28, 28))
        return [images]
