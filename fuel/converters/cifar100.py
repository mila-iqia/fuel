import os
import tarfile

import h5py
import numpy
import six
from six.moves import cPickle

from fuel.converters.base import fill_hdf5_file, check_exists

DISTRIBUTION_FILE = 'cifar-100-python.tar.gz'


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_cifar100(directory, output_directory,
                     output_filename='cifar100.hdf5'):
    """Converts the CIFAR-100 dataset to HDF5.

    Converts the CIFAR-100 dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CIFAR100`. The converted dataset is saved as
    'cifar100.hdf5'.

    This method assumes the existence of the following file:
    `cifar-100-python.tar.gz`

    Parameters
    ----------
    directory : str
        Directory in which the required input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'cifar100.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode="w")
    input_file = os.path.join(directory, 'cifar-100-python.tar.gz')
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
    train_coarse_labels = numpy.array(train['coarse_labels'],
                                      dtype=numpy.uint8)
    train_fine_labels = numpy.array(train['fine_labels'],
                                    dtype=numpy.uint8)

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
            ('train', 'coarse_labels', train_coarse_labels.reshape((-1, 1))),
            ('train', 'fine_labels', train_fine_labels.reshape((-1, 1))),
            ('test', 'features', test_features),
            ('test', 'coarse_labels', test_coarse_labels.reshape((-1, 1))),
            ('test', 'fine_labels', test_fine_labels.reshape((-1, 1))))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['coarse_labels'].dims[0].label = 'batch'
    h5file['coarse_labels'].dims[1].label = 'index'
    h5file['fine_labels'].dims[0].label = 'batch'
    h5file['fine_labels'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the CIFAR100 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar100` command.

    """
    return convert_cifar100
