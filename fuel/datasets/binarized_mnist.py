# -*- coding: utf-8 -*-
import logging
import os

import numpy

from fuel import config
from fuel.datasets import Dataset
from fuel.schemes import SequentialScheme
from fuel.utils import do_not_pickle_attributes

logger = logging.getLogger(__name__)


@do_not_pickle_attributes('features')
class BinarizedMNIST(Dataset):
    u"""The binarized, unlabeled MNIST dataset used for evaluating
    generative models (e.g. DBN, VAE and NADE).

    MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
    a database of handwritten digits. It is one of the most famous datasets
    in machine learning and consists of 60,000 training images and 10,000
    testing images. The images are grayscale and 28 x 28 pixels large.

    This particular version of the dataset is the one used in R.
    Salakhutdinov's DBN paper [DBN] as well as the VAE and NADE papers,
    and is accessible through Hugo Larochelle's public website [HUGO].

    The training set has further been split into a training and a
    validation set. All examples were binarized by sampling from a
    binomial distribution defined by the pixel values.

    .. [LBBH] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner,
       *Gradient-based learning applied to document recognition*,
       Proceedings of the IEEE, November 1998, 86(11):2278-2324.

    .. [DBN] Ruslan Salakhutdinov and Iain Murray, *On the Quantitative
       Analysis of Deep Belief Networks*, Proceedings of the 25th
       international conference on Machine learning, 2008, pp. 872-879.

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{train,valid,test}.amat

    .. todo::

       Right now this dataset always returns flattened images. In order to
       support e.g. convolutions and visualization, it needs to support the
       original 28 x 28 image format.

    Parameters
    ----------
    which_set : 'train' or 'valid' or 'test'
        Whether to load the training set (50,000 samples) or the validation
        set (10,000 samples) or the test set (10,000 samples).

    """
    provides_sources = ('features',)
    base_path = os.path.join(config.data_path, 'binarized_mnist')

    def __init__(self, which_set, **kwargs):
        if which_set not in ('train', 'valid', 'test'):
            raise ValueError("available splits are 'train', 'valid' and "
                             "'test'")
        self.num_examples = 50000 if which_set == 'train' else 10000
        self.default_scheme = SequentialScheme(self.num_examples, 1)
        super(BinarizedMNIST, self).__init__(**kwargs)

        self.which_set = which_set
        self.data_path = os.path.join(
            self.base_path, 'binarized_mnist_' + self.which_set + '.npy')

    def load(self):
        # If only the .amat file is avaiable, do the conversion
        if os.path.isfile(self.data_path):
            x = numpy.load(self.data_path).astype('float64')
        else:
            logger.warn("The faster .npy version of " +
                        "binarized_mnist_{} ".format(self.which_set) +
                        "isn't available, falling back to the .amat version.")
            x = numpy.loadtxt(self.data_path[:-3] + 'amat', dtype='float64')
        self.features = x

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError("BinarizedMNIST does not have a state")
        return self.filter_sources((self.features[request],))
