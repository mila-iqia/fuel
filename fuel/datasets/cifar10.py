import os
from collections import OrderedDict

import numpy
import six
from six.moves import cPickle

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('indexables')
class CIFAR10(IndexableDataset):
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
    folder = 'cifar10'
    files = {
        'train': [os.path.join('cifar-10-batches-py',
                               'data_batch_{}'.format(i))
                  for i in range(1, 6)],
        'test': ['cifar-10-batches-py/test_batch']
    }

    def __init__(self, which_set, flatten=True, **kwargs):
        if which_set not in ('train', 'test'):
            raise ValueError("CIFAR10 only has a train and test set")

        self.which_set = which_set
        self.flatten = flatten

        super(CIFAR10, self).__init__(OrderedDict(zip(self.provides_sources,
                                                      self._load_cifar10())),
                                      **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_cifar10())
                           if source in self.sources]

    def _load_cifar10(self):
        base_path = os.path.join(config.data_path, self.folder)
        num_examples = 50000 if self.which_set == 'train' else 10000
        image_shape = (3072,) if self.flatten else (3, 32, 32)
        images = numpy.zeros((num_examples,) + image_shape,
                             dtype=config.floatX)
        labels = numpy.zeros((num_examples, 1), dtype='uint8')
        for i, fname in enumerate(self.files[self.which_set]):
            with open(os.path.join(base_path, fname), 'rb') as f:
                if six.PY3:
                    batch = cPickle.load(f, encoding='latin1')
                else:
                    batch = cPickle.load(f)
                if not self.flatten:
                    batch['data'] = batch['data'].reshape((10000, 3, 32, 32))
                images[10000 * i:10000 * (i + 1)] = batch['data']
                labels[10000 * i:10000 * (i + 1)] = numpy.asarray(
                    batch['labels'], dtype='uint8')[:, None]
        return images, labels
