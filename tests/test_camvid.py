import os

import h5py
import numpy
from numpy.testing import assert_equal

from fuel import config
from fuel.datasets import H5PYDataset, Camvid


def test_camvid():
    data_path = config.data_path
    try:
        config.data_path = '.'
        f = h5py.File('camvid.hdf5', 'w')
        f['features'] = numpy.arange(
            10 * 3 * 360 * 480, dtype='uint8').reshape((10, 3, 360, 480))
        f['targets'] = numpy.arange(
            10 * 360 * 480, dtype='uint8').reshape((10, 360, 480))
        split_dict = {'train': {'features': (0, 6), 'targets': (0, 6)},
                      'valid': {'features': (6, 8), 'targets': (6, 8)},
                      'test': {'features': (8, 10), 'targets': (8, 10)}}
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.close()
        dataset = Camvid(which_sets=('train',))
        assert_equal(dataset.filename, 'camvid.hdf5')
    finally:
        config.data_path = data_path
os.remove('camvid.hdf5')
