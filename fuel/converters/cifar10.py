import os
import tarfile

import h5py
import numpy
import six
from six.moves import range, cPickle

from fuel.converters.base import fill_hdf5_file, check_exists

DISTRIBUTION_FILE = 'cifar-10-python.tar.gz'


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_cifar10(directory, output_directory,
                    output_filename='cifar10.hdf5'):
    """Converts the CIFAR-10 dataset to HDF5.

    Converts the CIFAR-10 dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CIFAR10`. The converted dataset is saved as
    'cifar10.hdf5'.

    It assumes the existence of the following file:

    * `cifar-10-python.tar.gz`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'cifar10.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    input_file = os.path.join(directory, DISTRIBUTION_FILE)
    tar_file = tarfile.open(input_file, 'r:gz')

    train_batches = []
    for batch in range(1, 6):
        file = tar_file.extractfile(
            'cifar-10-batches-py/data_batch_%d' % batch)
        try:
            if six.PY3:
                array = cPickle.load(file, encoding='latin1')
            else:
                array = cPickle.load(file)
            train_batches.append(array)
        finally:
            file.close()

    train_features = numpy.concatenate(
        [batch['data'].reshape(batch['data'].shape[0], 3, 32, 32)
            for batch in train_batches])
    train_labels = numpy.concatenate(
        [numpy.array(batch['labels'], dtype=numpy.uint8)
            for batch in train_batches])
    train_labels = numpy.expand_dims(train_labels, 1)

    file = tar_file.extractfile('cifar-10-batches-py/test_batch')
    try:
        if six.PY3:
            test = cPickle.load(file, encoding='latin1')
        else:
            test = cPickle.load(file)
    finally:
        file.close()

    test_features = test['data'].reshape(test['data'].shape[0],
                                         3, 32, 32)
    test_labels = numpy.array(test['labels'], dtype=numpy.uint8)
    test_labels = numpy.expand_dims(test_labels, 1)

    data = (('train', 'features', train_features),
            ('train', 'targets', train_labels),
            ('test', 'features', test_features),
            ('test', 'targets', test_labels))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    return convert_cifar10
