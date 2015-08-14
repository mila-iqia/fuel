import gzip
import os
import struct

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file, check_exists

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

ALL_FILES = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]


@check_exists(required_files=ALL_FILES)
def convert_mnist(directory, output_directory, output_filename=None,
                  dtype=None):
    """Converts the MNIST dataset to HDF5.

    Converts the MNIST dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.MNIST`. The converted dataset is
    saved as 'mnist.hdf5'.

    This method assumes the existence of the following files:
    `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`
    `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`

    It assumes the existence of the following files:

    * `train-images-idx3-ubyte.gz`
    * `train-labels-idx1-ubyte.gz`
    * `t10k-images-idx3-ubyte.gz`
    * `t10k-labels-idx1-ubyte.gz`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to `None`, in which case a name
        based on `dtype` will be used.
    dtype : str, optional
        Either 'float32', 'float64', or 'bool'. Defaults to `None`,
        in which case images will be returned in their original
        unsigned byte format.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    if not output_filename:
        if dtype:
            output_filename = 'mnist_{}.hdf5'.format(dtype)
        else:
            output_filename = 'mnist.hdf5'
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')

    train_feat_path = os.path.join(directory, TRAIN_IMAGES)
    train_features = read_mnist_images(train_feat_path, dtype)
    train_lab_path = os.path.join(directory, TRAIN_LABELS)
    train_labels = read_mnist_labels(train_lab_path)
    test_feat_path = os.path.join(directory, TEST_IMAGES)
    test_features = read_mnist_images(test_feat_path, dtype)
    test_lab_path = os.path.join(directory, TEST_LABELS)
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
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the MNIST dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    subparser.add_argument(
        "--dtype", help="dtype to save to; by default, images will be " +
        "returned in their original unsigned byte format",
        choices=('float32', 'float64', 'bool'), type=str, default=None)
    return convert_mnist


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
    labels : :class:`~numpy.ndarray`, shape (nlabels, 1)
        A one-dimensional unsigned byte array containing the
        labels as integers.

    """
    with gzip.open(filename, 'rb') as f:
        magic, _ = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Wrong magic number reading MNIST label file")
        array = numpy.frombuffer(f.read(), dtype='uint8')
    array = array.reshape(array.size, 1)
    return array
