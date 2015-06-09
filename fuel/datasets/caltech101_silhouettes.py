# -*- coding: utf-8 -*-
import os

from fuel import config
from fuel.datasets import H5PYDataset


class CalTech101Silhouettes(H5PYDataset):
    u"""CalTech 101 Silhouettes dataset.

    This dataset provides the `split1` train/validation/test split of the
    CalTech101 Silhouette dataset prepared by Benjamin M. Marlin [MARLIN].

    This class provides both the 16x16 and the 28x28 pixel sized version.

    .. [MARLIN] https://people.cs.umass.edu/~marlin/data.shtml

    Parameters
    ----------
    which_set : 'train' or 'valid' or 'test'
        Whether to load the training set (4,100 samples) or the validation
        set (2,264 samples) or the test set (2,307 samples).
    size : int
        Either 16 or 28 to select the 16x16 or 28x28 pixels version
        of the dataset (default: 28).

    """
    def __init__(self, which_set, size=28, load_in_memory=True, **kwargs):
        if size not in (16, 28):
            raise ValueError('size must be 16 or 28')

        self.filename = 'caltech101_silhouettes{}.hdf5'.format(size)
        super(CalTech101Silhouettes, self).__init__(
            self.data_path, which_set=which_set,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
