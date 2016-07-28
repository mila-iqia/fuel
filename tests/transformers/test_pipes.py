from fuel.transformers import Mapping
from fuel.datasets import MNIST
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream

from numpy.testing import assert_allclose

dataset = MNIST(('train',))


def foo(batch):
    return batch


def test_pipes():
    stream = (DataStream.default_stream(
        dataset, iteration_scheme=ConstantScheme(10)) > Mapping(mapping=foo) |
              Mapping(mapping=foo))
    stream2 = Mapping(
        Mapping(
            DataStream.default_stream(dataset,
                                      iteration_scheme=ConstantScheme(10)),
            mapping=foo),
        mapping=foo)

    dt = next(stream.get_epoch_iterator())
    dt2 = next(stream2.get_epoch_iterator())

    assert_allclose(dt[0], dt2[0])
