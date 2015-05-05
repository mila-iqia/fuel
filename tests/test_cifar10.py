import numpy
from numpy.testing import assert_raises

from fuel.datasets import CIFAR10


def test_cifar10():
    train = CIFAR10('train', flatten=('features',),
                    load_in_memory=False)
    assert train.num_examples == 50000
    handle = train.open()
    features, targets = train.get_data(handle, slice(49000, 50000))
    assert features.shape == (1000, 3072)
    assert targets.shape == (1000,)
    train.close(handle)

    test = CIFAR10('test', sources=('targets',), load_in_memory=False)
    handle = test.open()
    targets, = test.get_data(handle, slice(0, 100))
    assert targets.shape == (100,)
    test.close(handle)

    test = CIFAR10('test', load_in_memory=False)
    handle = test.open()
    features, targets = test.get_data(handle, slice(0, 100))
    assert features.shape == (100, 3, 32, 32)
    assert targets.shape == (100,)
    assert features.dtype.kind == 'f'
    assert targets.dtype == numpy.uint8
    test.close(handle)

    assert_raises(ValueError, CIFAR10, 'valid')
