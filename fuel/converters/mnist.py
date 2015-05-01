import gzip
import os
import struct

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file
MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


def mnist(input_directory, save_path, dtype=None):
    """Converts the MNIST dataset to HDF5.

    Converts the MNIST dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.MNIST`. The converted dataset is
    saved as 'mnist.hdf5'.

    This method assumes the existence of the following files:
    `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`
    `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`

    Parameters
    ----------
    input_directory : str
        Directory in which the required input files reside.
    save_path : str
        Where to save the converted dataset.
    dtype : 'float32', 'float64', or 'bool'
        If unspecified, images will be returned in their original
        unsigned byte format.

    """
    h5file = h5py.File(save_path, mode="w")
    train_feat_path = os.path.join(input_directory,
                                   'train-images-idx3-ubyte.gz')
    train_features = read_mnist_images(train_feat_path, dtype)
    train_lab_path = os.path.join(input_directory,
                                  'train-labels-idx1-ubyte.gz')
    train_labels = read_mnist_labels(train_lab_path)
    test_feat_path = os.path.join(input_directory,
                                  't10k-images-idx3-ubyte.gz')
    test_features = read_mnist_images(test_feat_path, dtype)
    test_lab_path = os.path.join(input_directory,
                                 't10k-labels-idx1-ubyte.gz')
    test_labels = read_mnist_labels(test_lab_path)
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

    h5file.flush()
    h5file.close()


def read_mnist_images(filename, dtype=None):
    """Read MNIST images from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read images.

    dtype : 'float32', 'float64', or 'bool'
        If unspecified, images will be returned in their original
        unsigned byte format.

    Returns
    -------
    images : :class:`~numpy.ndarray`, shape (n_images, 1, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----
    If the dtype provided was Boolean, the resulting array will
    be Boolean with `True` if the corresponding pixel had a value
    greater than or equal to 128, `False` otherwise.

    If the dtype provided was a float dtype, the values will be mapped to
    the unit interval [0, 1], with pixel values that were 255 in the
    original unsigned byte representation equal to 1.0.

    """
    with gzip.open(filename, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError("Wrong magic number reading MNIST image file")
        array = numpy.frombuffer(f.read(), dtype='uint8')
        array = array.reshape((number, 1, rows, cols))
    if dtype:
        dtype = numpy.dtype(dtype)

        if dtype.kind == 'b':
            # If the user wants Booleans, threshold at half the range.
            array = array >= 128
        elif dtype.kind == 'f':
            # Otherwise, just convert.
            array = array.astype(dtype)
            array /= 255.
        else:
            raise ValueError("Unknown dtype to convert MNIST to")
    return array


def read_mnist_labels(filename):
    """Read MNIST labels from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read labels.

    Returns
    -------
    labels : :class:`~numpy.ndarray`, shape (nlabels,)
        A one-dimensional unsigned byte array containing the
        labels as integers.

    """
    with gzip.open(filename, 'rb') as f:
        magic, _ = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Wrong magic number reading MNIST label file")
        array = numpy.frombuffer(f.read(), dtype='uint8')
    return array
