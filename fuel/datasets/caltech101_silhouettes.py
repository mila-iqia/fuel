# -*- coding: utf-8 -*-
from fuel.utils import find_in_data_path
from fuel.datasets import H5PYDataset


class CalTech101Silhouettes(H5PYDataset):
    u"""CalTech 101 Silhouettes dataset.

    This dataset provides the `split1` train/validation/test split of the
    CalTech101 Silhouette dataset prepared by Benjamin M. Marlin [MARLIN].

    This class provides both the 16x16 and the 28x28 pixel sized version.
    The 16x16 version contains 4082 examples in the training set, 2257
    examples in the validation set and 2302 examples in the test set. The
    28x28 version contains 4100, 2264 and 2307 examples in the train, valid
    and test set.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'valid' and 'test'.
    size : {16, 28}
        Either 16 or 28 to select the 16x16 or 28x28 pixels version
        of the dataset (default: 28).

    """
    def __init__(self, which_sets, size=28, load_in_memory=True, **kwargs):
        if size not in (16, 28):
            raise ValueError('size must be 16 or 28')

        self.filename = 'caltech101_silhouettes{}.hdf5'.format(size)
        super(CalTech101Silhouettes, self).__init__(
            self.data_path, which_sets=which_sets,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return find_in_data_path(self.filename)
