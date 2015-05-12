import os
import tarfile

import h5py
import numpy
import six
from six.moves import range, cPickle

from fuel.converters.base import fill_hdf5_file


def cifar100(input_directory, save_path):
    """Converts the CIFAR-100 dataset to HDF5.

    Converts the CIFAR-100 dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CIFAR100`. The converted dataset is saved as
    'cifar100.hdf5'.

    This method assumes the existence of the following file:
    `cifar-100-python.tar.gz`

    Parameters
    ----------
    input_directory : str
        Directory in which the required input files reside.
    save_path : str
        Where to save the converted dataset.

    """
    h5file = h5py.File(save_path, mode="w")
    input_file = os.path.join(input_directory, 'cifar-100-python.tar.gz')
    tar_file = tarfile.open(input_file, 'r:gz')

    file = tar_file.extractfile('cifar-100-python/train')
    try:
        if six.PY3:
            train = cPickle.load(file, encoding='latin1')
        else:
            train = cPickle.load(file)
    finally:
        file.close()

    train_features = train['data'].reshape(train['data'].shape[0],
                                         3, 32, 32)
    train_coarse_labels = numpy.array(train['coarse_labels'], dtype=numpy.uint8)
    train_fine_labels = numpy.array(train['fine_labels'], dtype=numpy.uint8)

    file = tar_file.extractfile('cifar-100-python/test')
    try:
        if six.PY3:
            test = cPickle.load(file, encoding='latin1')
        else:
            test = cPickle.load(file)
    finally:
        file.close()

    test_features = test['data'].reshape(test['data'].shape[0],
                                         3, 32, 32)                                         
    test_coarse_labels = numpy.array(test['coarse_labels'], dtype=numpy.uint8)
    test_fine_labels = numpy.array(test['fine_labels'], dtype=numpy.uint8)

    data = (('train', 'features', train_features),
            ('train', 'coarse_labels', train_coarse_labels),
            ('train', 'fine_labels', train_fine_labels),
            ('test', 'features', test_features),
            ('test', 'coarse_labels', test_coarse_labels),
            ('test', 'fine_labels', test_fine_labels))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['coarse_labels'].dims[0].label = 'batch'
    h5file['fine_labels'].dims[0].label = 'batch'

    h5file.flush()
    h5file.close()
