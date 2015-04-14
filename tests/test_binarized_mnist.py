import os

from numpy.testing import assert_raises

from fuel import config
from fuel.datasets import BinarizedMNIST
from tests import skip_if_not_available


def test_binarized_mnist_train():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST('train', load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert dataset.num_examples == 50000
    dataset.close(handle)


def test_binarized_mnist_valid():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST('valid', load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert dataset.num_examples == 10000
    dataset.close(handle)


def test_binarized_mnist_test():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST('test', load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert dataset.num_examples == 10000
    dataset.close(handle)


def test_binarized_mnist_invalid_split():
    assert_raises(ValueError, BinarizedMNIST, 'dummy')


def test_binarized_mnist_data_path():
    assert BinarizedMNIST('train').data_path == os.path.join(
        config.data_path, 'binarized_mnist.hdf5')
