from numpy.testing import assert_raises

from fuel.datasets import BinarizedMNIST
from tests import skip_if_not_available


def test_binarized_mnist():
    skip_if_not_available(datasets=['binarized_mnist'])

    mnist_train = BinarizedMNIST('train')
    handle = mnist_train.open()
    data = mnist_train.get_data(handle, slice(0, 50000))[0]
    assert data.shape == (50000, 1, 28, 28)
    assert mnist_train.num_examples == 50000
    mnist_train.close(handle)

    mnist_valid = BinarizedMNIST('valid')
    handle = mnist_valid.open()
    data = mnist_valid.get_data(handle, slice(0, 10000))[0]
    assert data.shape == (10000, 1, 28, 28)
    assert mnist_valid.num_examples == 10000
    mnist_valid.close(handle)

    mnist_test = BinarizedMNIST('test')
    handle = mnist_test.open()
    data = mnist_test.get_data(handle, slice(0, 10000))[0]
    assert data.shape == (10000, 1, 28, 28)
    assert mnist_test.num_examples == 10000
    mnist_test.close(handle)

    assert_raises(ValueError, BinarizedMNIST, 'dummy')

    mnist_test_flattened = BinarizedMNIST(
        'test', flatten=True, load_in_memory=True)
    handle = mnist_test_flattened.open()
    data = mnist_test_flattened.get_data(handle, slice(0, 10000))[0]
    assert data.shape == (10000, 784)
    mnist_test_flattened.close(handle)
