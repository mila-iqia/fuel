import hashlib

from numpy.testing import assert_raises, assert_equal

from fuel.datasets import BinarizedMNIST
from tests import skip_if_not_available


def test_binarized_mnist_train():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST(('train',), load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert hashlib.md5(data).hexdigest() == '0922fefc9a9d097e3b086b89107fafce'
    assert dataset.num_examples == 50000
    dataset.close(handle)


def test_binarized_mnist_valid():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST(('valid',), load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert hashlib.md5(data).hexdigest() == '65e8099613162b3110a7618037011617'
    assert dataset.num_examples == 10000
    dataset.close(handle)


def test_binarized_mnist_test():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST(('test',), load_in_memory=False)
    handle = dataset.open()
    data, = dataset.get_data(handle, slice(0, 10))
    assert data.dtype == 'uint8'
    assert data.shape == (10, 1, 28, 28)
    assert hashlib.md5(data).hexdigest() == '0fa539ed8cb008880a61be77f744f06a'
    assert dataset.num_examples == 10000
    dataset.close(handle)


def test_binarized_mnist_axes():
    skip_if_not_available(datasets=['binarized_mnist.hdf5'])

    dataset = BinarizedMNIST(('train',), load_in_memory=False)
    assert_equal(dataset.axis_labels['features'],
                 ('batch', 'channel', 'height', 'width'))


def test_binarized_mnist_invalid_split():
    assert_raises(ValueError, BinarizedMNIST, ('dummy',))
