import os

import fuel
import h5py
import numpy

default_directory = os.path.join(fuel.config.data_path, 'binarized_mnist')
default_save_path = os.path.join(default_directory, 'binarized_mnist.hdf5')


def binarized_mnist(directory=None, save_path=None):
    """Converts the binarized MNIST dataset to HDF5.

    Converts the binarized MNIST dataset used in R. Salakhutdinov's DBN
    paper [DBN] to an HDF5 dataset compatible with
    :class:`fuel.datasets.binarized_mnist.BinarizedMNIST`.

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
    directory : str, optional
        Base directory in which the required input files reside. Defaults
        to `None`, in which case `'$FUEL_DATA_PATH/binarized_mnist'` is
        used.
    save_path : str, optional
        Where to save the converted dataset. Defaults to `None`, in which
        case `'$FUEL_DATA_PATH/binarized_mnist/binarized_mnist.hdf5'` is
        used.

    """
    if directory is None:
        directory = default_directory
    if save_path is None:
        save_path = default_save_path

    train_set = numpy.loadtxt(
        os.path.join(directory, 'binarized_mnist_train.amat'))
    valid_set = numpy.loadtxt(
        os.path.join(directory, 'binarized_mnist_valid.amat'))
    test_set = numpy.loadtxt(
        os.path.join(directory, 'binarized_mnist_test.amat'))

    f = h5py.File(save_path, mode="w")

    features = f.create_dataset('features', (70000, 1, 28, 28), dtype='uint8')
    features[...] = numpy.vstack([train_set.reshape((-1, 1, 28, 28)),
                                  valid_set.reshape((-1, 1, 28, 28)),
                                  test_set.reshape((-1, 1, 28, 28))])
    f.attrs['train'] = [0, 50000]
    f.attrs['valid'] = [50000, 60000]
    f.attrs['test'] = [60000, 70000]

    f.flush()
    f.close()
