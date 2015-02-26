import numpy
from numpy.testing import assert_raises

from fuel.datasets import BinarizedMNIST


def test_mnist():
    mnist_train = BinarizedMNIST('train')
    assert len(mnist_train.features) == 50000
    assert mnist_train.num_examples == 50000
    mnist_valid = BinarizedMNIST('valid')
    assert len(mnist_valid.features) == 10000
    assert mnist_valid.num_examples == 10000
    mnist_test = BinarizedMNIST('test')
    assert len(mnist_test.features) == 10000
    assert mnist_test.num_examples == 10000

    first_feature, = mnist_train.get_data(request=[0])
    assert first_feature.shape == (1, 784)
    assert first_feature.dtype.kind == 'f'

    assert_raises(ValueError, BinarizedMNIST, 'dummy')
