import os

import h5py
from numpy.testing import assert_equal

from fuel import config
from fuel.datasets import SVHN


def test_svhn():
    try:
        f = h5py.File('svhn_format_2.hdf5', 'w')
        f.close()
        dataset = SVHN(which_format=2, which_set='train')
        assert_equal(dataset.data_path,
                     os.path.join(config.data_path, 'svhn_format_2.hdf5'))
    finally:
        os.remove('svhn_format_2.hdf5')
