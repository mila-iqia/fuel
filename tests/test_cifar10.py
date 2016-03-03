import numpy
from numpy.testing import assert_raises

from fuel import config
from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


def test_cifar10():
    train = CIFAR10(('train',), load_in_memory=False)
    assert train.num_examples == 50000
    handle = train.open()
    features, targets = train.get_data(handle, slice(49990, 50000))
    assert features.shape == (10, 3, 32, 32)
    assert targets.shape == (10, 1)
    train.close(handle)

    test = CIFAR10(('test',), load_in_memory=False)
    handle = test.open()
    features, targets = test.get_data(handle, slice(0, 10))
    assert features.shape == (10, 3, 32, 32)
    assert targets.shape == (10, 1)
    assert features.dtype == numpy.uint8
    assert targets.dtype == numpy.uint8
    test.close(handle)

    stream = DataStream.default_stream(
        test, iteration_scheme=SequentialScheme(10, 10))
    data = next(stream.get_epoch_iterator())[0]
    assert data.min() >= 0.0 and data.max() <= 1.0
    assert data.dtype == config.floatX

    assert_raises(ValueError, CIFAR10, ('valid',))

    assert_raises(ValueError, CIFAR10,
                  ('train',), subset=slice(50000, 60000))
