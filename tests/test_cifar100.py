import numpy
from numpy.testing import assert_raises

from fuel import config
from fuel.datasets import CIFAR100
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


def test_cifar100():
    train = CIFAR100(('train',), load_in_memory=False)
    assert train.num_examples == 50000
    handle = train.open()
    coarse_labels, features, fine_labels = train.get_data(handle,
                                                          slice(49990, 50000))

    assert features.shape == (10, 3, 32, 32)
    assert coarse_labels.shape == (10, 1)
    assert fine_labels.shape == (10, 1)
    train.close(handle)

    test = CIFAR100(('test',), load_in_memory=False)
    handle = test.open()
    coarse_labels, features, fine_labels = test.get_data(handle,
                                                         slice(0, 10))

    assert features.shape == (10, 3, 32, 32)
    assert coarse_labels.shape == (10, 1)
    assert fine_labels.shape == (10, 1)

    assert features.dtype == numpy.uint8
    assert coarse_labels.dtype == numpy.uint8
    assert fine_labels.dtype == numpy.uint8

    test.close(handle)

    stream = DataStream.default_stream(
        test, iteration_scheme=SequentialScheme(10, 10))
    data = next(stream.get_epoch_iterator())[1]

    assert data.min() >= 0.0 and data.max() <= 1.0
    assert data.dtype == config.floatX

    assert_raises(ValueError, CIFAR100, ('valid',))
