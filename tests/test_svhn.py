import os

import h5py
import numpy
from numpy.testing import assert_equal

from fuel import config
from fuel.datasets import H5PYDataset, SVHN


def test_svhn():
    data_path = config.data_path
    try:
        config.data_path = '.'
        f = h5py.File('svhn_format_2.hdf5', 'w')
        f['features'] = numpy.arange(100, dtype='uint8').reshape((10, 10))
        f['targets'] = numpy.arange(10, dtype='uint8').reshape((10, 1))
        split_dict = {'train': {'features': (0, 8), 'targets': (0, 8)},
                      'test': {'features': (8, 10), 'targets': (8, 10)}}
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.close()
        dataset = SVHN(which_format=2, which_sets=('train',))
        assert_equal(dataset.filename, 'svhn_format_2.hdf5')
    finally:
        config.data_path = data_path
        os.remove('svhn_format_2.hdf5')
