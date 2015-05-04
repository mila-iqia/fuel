import os

from fuel import config
from fuel.datasets import H5PYDataset


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
        if which_set not in ('train', 'test'):
            raise ValueError("CIFAR10 only has a train and test set")

        kwargs.setdefault('load_in_memory', True)
        self.which_set = which_set

        super(CIFAR10, self).__init__(self.data_path, which_set,
                                      **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)

    def get_data(self, state=None, request=None):
        rval = list(super(CIFAR10, self).get_data(state=state,
                                                  request=request))
        if 'features' in self.sources:
            i = self.sources.index('features')
            rval[i] = (rval[i] / 255.0).astype(config.floatX)
        return tuple(rval)
