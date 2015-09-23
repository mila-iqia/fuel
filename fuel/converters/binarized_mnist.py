import os

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file, check_exists


TRAIN_FILE = 'binarized_mnist_train.amat'
VALID_FILE = 'binarized_mnist_valid.amat'
TEST_FILE = 'binarized_mnist_test.amat'

ALL_FILES = [TRAIN_FILE, VALID_FILE, TEST_FILE]


@check_exists(required_files=ALL_FILES)
def convert_binarized_mnist(directory, output_directory,
                            output_filename='binarized_mnist.hdf5'):
    """Converts the binarized MNIST dataset to HDF5.

    Converts the binarized MNIST dataset used in R. Salakhutdinov's DBN
    paper [DBN] to an HDF5 dataset compatible with
    :class:`fuel.datasets.BinarizedMNIST`. The converted dataset is
    saved as 'binarized_mnist.hdf5'.

    This method assumes the existence of the files
    `binarized_mnist_{train,valid,test}.amat`, which are accessible
    through Hugo Larochelle's website [HUGO].

    .. [DBN] Ruslan Salakhutdinov and Iain Murray, *On the Quantitative
       Analysis of Deep Belief Networks*, Proceedings of the 25th
       international conference on Machine learning, 2008, pp. 872-879.

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'binarized_mnist.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')

    train_set = numpy.loadtxt(
        os.path.join(directory, TRAIN_FILE)).reshape(
            (-1, 1, 28, 28)).astype('uint8')
    valid_set = numpy.loadtxt(
        os.path.join(directory, VALID_FILE)).reshape(
            (-1, 1, 28, 28)).astype('uint8')
    test_set = numpy.loadtxt(
        os.path.join(directory, TEST_FILE)).reshape(
            (-1, 1, 28, 28)).astype('uint8')
    data = (('train', 'features', train_set),
            ('valid', 'features', valid_set),
            ('test', 'features', test_set))
    fill_hdf5_file(h5file, data)
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the binarized MNIST dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `binarized_mnist` command.

    """
    return convert_binarized_mnist
