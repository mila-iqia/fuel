import os

import h5py
import numpy
from numpy.testing import assert_equal

from fuel import config
from fuel.datasets import H5PYDataset, CelebA


def test_celeba():
    data_path = config.data_path
    try:
        config.data_path = '.'
        f = h5py.File('celeba_64.hdf5', 'w')
        f['features'] = numpy.arange(
            10 * 3 * 64 * 64, dtype='uint8').reshape((10, 3, 64, 64))
        f['targets'] = numpy.arange(
            10 * 40, dtype='uint8').reshape((10, 40))
        split_dict = {'train': {'features': (0, 6), 'targets': (0, 6)},
                      'valid': {'features': (6, 8), 'targets': (6, 8)},
                      'test': {'features': (8, 10), 'targets': (8, 10)}}
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.close()
        dataset = CelebA(which_format='64', which_sets=('train',))
        assert_equal(dataset.filename, 'celeba_64.hdf5')
    finally:
        config.data_path = data_path
        os.remove('celeba_64.hdf5')
