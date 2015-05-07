import os

from fuel import config
from fuel.datasets import H5PYDataset
from fuel.transformers import ForceFloatX, ScaleAndShift


class CIFAR10(H5PYDataset):
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

    """
    provides_sources = ('features', 'targets')
    filename = 'cifar10.hdf5'

    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(CIFAR10, self).__init__(self.data_path, which_set, **kwargs)
        self.default_transformers += (
            (ScaleAndShift, [1 / 255.0, 0], {'which_sources': ('features',)}),
            (ForceFloatX, [], {}))

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
