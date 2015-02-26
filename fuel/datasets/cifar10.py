# -*- coding: utf-8 -*-
import os

import numpy
import six
from six.moves import cPickle, xrange

from fuel import config
from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes
from fuel.schemes import SequentialScheme


@do_not_pickle_attributes('features', 'targets')
class CIFAR10(Dataset):
    """The CIFAR10 dataset of natural images.

    This dataset is a labeled subset of the ``80 million tiny images''
    dataset [TINY]. It consists of 60,000 32 x 32 colour images in 10
    classes, with 6,000 images per class. There are 50,000 training
    images and 10,000 test images [CIFAR10].

    .. [TINY] Antonio Torralba, Rob Fergus and William T. Freeman,
       *80 million tiny images: a large dataset for non-parametric
       object and scene recognition*, Pattern Analysis and Machine
       Intelligence, IEEE Transactions on 30.11 (2008): 1958-1970.

    .. [CIFAR10] Alex Krizhevsky, *Learning Multiple Layers of Features
       from Tiny Images*, technical report, 2009.

    .. todo::

       Right now this dataset always returns flattened images. In order to
       support e.g. convolutions and visualization, it needs to support the
       original 32 x 32 x 3 image format.

    Parameters
    ----------
    which_set : 'train' or 'test'
        Whether to load the training set (50,000 samples) or the test set
        (10,000 samples). Note that CIFAR10 does not have a validation
        set; usually you will create your own training/validation split
        using the start and stop arguments.
    start : int, optional
        The first example to load
    stop : int, optional
        The last example to load

    """
    provides_sources = ('features', 'targets')

    def __init__(self, which_set, start=None, stop=None, **kwargs):
        if which_set not in ('train', 'test'):
            raise ValueError("CIFAR10 only has a train and test set")
        if stop is None:
            stop = 50000 if which_set == "train" else 10000
        if start is None:
            start = 0
        self.num_examples = stop - start
        self.default_scheme = SequentialScheme(self.num_examples, 1)
        super(CIFAR10, self).__init__(**kwargs)

        self.which_set = which_set
        self.start = start
        self.stop = stop

    def load(self):
        base_path = os.path.join(
            config.data_path, 'cifar10', 'cifar-10-batches-py')
        if self.which_set == 'train':
            fnames = ['data_batch_%i' % i for i in xrange(1, 6)]
            x = numpy.zeros((50000, 3072), dtype='float64')
            y = numpy.zeros((50000, 1), dtype='uint8')
            for i, fname in enumerate(fnames):
                with open(os.path.join(base_path, fname), 'rb') as f:
                    if six.PY3:
                        data = cPickle.load(f, encoding='latin1')
                    else:
                        data = cPickle.load(f)
                    x[10000 * i: 10000 * (i + 1)] = data['data']
                    y[10000 * i: 10000 * (i + 1)] = numpy.asarray(
                        data['labels'], dtype='uint8')[:, numpy.newaxis]
            x = x[self.start: self.stop]
            y = y[self.start: self.stop]
        elif self.which_set == 'test':
            with open(os.path.join(base_path, 'test_batch'), 'rb') as f:
                if six.PY3:
                    data = cPickle.load(f, encoding='latin1')
                else:
                    data = cPickle.load(f)
                x = data['data'].astype('float64')[self.start: self.stop]
                y = numpy.asarray(
                    data['labels'], dtype='uint8')[self.start: self.stop,
                                                   numpy.newaxis]
        self.features = x
        self.targets = y

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError("CIFAR10 does not have a state")
        return self.filter_sources((self.features[request],
                                    self.targets[request]))
