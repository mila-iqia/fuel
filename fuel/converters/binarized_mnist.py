import os

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file


def binarized_mnist(input_directory, save_path):
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

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    input_directory : str
        Directory in which the required input files reside.
    save_path : str
        Where to save the converted dataset.

    """
    h5file = h5py.File(save_path, mode="w")
    train_set = numpy.loadtxt(
        os.path.join(input_directory, 'binarized_mnist_train.amat')).reshape(
            (-1, 1, 28, 28))
    valid_set = numpy.loadtxt(
        os.path.join(input_directory, 'binarized_mnist_valid.amat')).reshape(
            (-1, 1, 28, 28))
    test_set = numpy.loadtxt(
        os.path.join(input_directory, 'binarized_mnist_test.amat')).reshape(
            (-1, 1, 28, 28))
    data = ((train_set, valid_set, test_set),)
    source_names = ('features',)
    shapes = ((70000, 1, 28, 28),)
    dtypes = ('uint8',)
    split_names = ('train', 'valid', 'test')
    splits = ((0, 50000), (50000, 60000), (60000, 70000))

    fill_hdf5_file(
        h5file, data, source_names, shapes, dtypes, split_names, splits)

    h5file.flush()
    h5file.close()
