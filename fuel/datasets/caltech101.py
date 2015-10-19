# -*- coding: utf-8 -*-
from fuel.utils import find_in_data_path
from fuel.datasets import H5PYDataset


class CalTech101(H5PYDataset):
    u"""CalTech 101 dataset.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'valid' and 'test'.

    """
    def __init__(self, which_sets, load_in_memory=True, **kwargs):
        self.filename = 'caltech101.hdf5'
        super(CalTech101, self).__init__(
            self.data_path, which_sets=which_sets,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return find_in_data_path(self.filename)
