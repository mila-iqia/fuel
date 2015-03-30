import os

from numpy.testing import assert_raises

from fuel import config
from fuel.datasets import BinarizedMNIST
from tests import skip_if_not_available


def test_binarized_mnist_train():
    skip_if_not_available(datasets=['binarized_mnist'])

    mnist_train = BinarizedMNIST('train')
    handle = mnist_train.open()
    data = mnist_train.get_data(handle, slice(0, 50000))[0]
    assert data.shape == (50000, 1, 28, 28)
    assert mnist_train.num_examples == 50000
    mnist_train.close(handle)


def test_binarized_mnist_valid():
    skip_if_not_available(datasets=['binarized_mnist'])

    mnist_valid = BinarizedMNIST('valid')
    handle = mnist_valid.open()
    data = mnist_valid.get_data(handle, slice(0, 10000))[0]
    assert data.shape == (10000, 1, 28, 28)
    assert mnist_valid.num_examples == 10000
    mnist_valid.close(handle)


def test_binarized_mnist_test():
    skip_if_not_available(datasets=['binarized_mnist'])

    mnist_test = BinarizedMNIST('test')
    handle = mnist_test.open()
    data = mnist_test.get_data(handle, slice(0, 10000))[0]
    assert data.shape == (10000, 1, 28, 28)
    assert mnist_test.num_examples == 10000
    mnist_test.close(handle)


def test_binarized_mnist_invalid_split():
    assert_raises(ValueError, BinarizedMNIST, 'dummy')


def test_binarized_mnist_data_path():
    assert BinarizedMNIST('train').data_path == os.path.join(
        config.data_path, 'binarized_mnist.hdf5')
