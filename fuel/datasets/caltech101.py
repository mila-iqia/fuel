# -*- coding: utf-8 -*-
from fuel.utils import find_in_data_path
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX


class CalTech101(H5PYDataset):
    u"""CalTech 101 dataset.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test'.

    """
    filename = 'caltech101.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features'),)

    def __init__(self, which_sets, load_in_memory=True, **kwargs):
        super(CalTech101, self).__init__(file_or_path=self.data_path,
                                         which_sets=which_sets,
                                         load_in_memory=load_in_memory,
                                         **kwargs)

    @property
    def data_path(self):
        return find_in_data_path(self.filename)
