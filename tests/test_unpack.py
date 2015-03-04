import numpy

from fuel.datasets import IterableDataset
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import Batch, Unpack


def test_unpack_transformer():
    data = range(10)
    stream = DataStream(IterableDataset(data))
    stream = Batch(stream, iteration_scheme=ConstantScheme(2))
    stream = Unpack(stream)
    epoch = stream.get_epoch_iterator()
    for i, v in enumerate(epoch):
        assert numpy.shape(v)[0] == 1
        assert v[0] == i
