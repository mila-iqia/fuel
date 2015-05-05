from collections import OrderedDict

import numpy
from numpy.testing import assert_raises
from picklable_itertools import repeat
from six.moves import zip, range

from fuel.datasets import IterableDataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import BatchSizeScheme, ConstantScheme
from fuel.transformers import Mapping


def test_dataset():
    data = [1, 2, 3]
    # The default stream requests an example at a time
    stream = DataStream(IterableDataset(data))
    epoch = stream.get_epoch_iterator()
    assert list(epoch) == list(zip(data))

    # Check if iterating over multiple epochs works
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == list(zip(data))

    # Check whether the returning as a dictionary of sources works
    assert next(stream.get_epoch_iterator(as_dict=True)) == {"data": 1}


def test_dataset_default_transformer():
    class DoublingDataset(IterableDataset):
        def apply_default_transformer(self, stream):
            return Mapping(
                stream, lambda sources: tuple(2 * s for s in sources))

    dataset = DoublingDataset([1, 2, 3])
    stream = dataset.apply_default_transformer(DataStream(dataset))
    epoch = stream.get_epoch_iterator()
    assert list(epoch) == [(2,), (4,), (6,)]


def test_dataset_no_axis_labels():
    dataset = IterableDataset(numpy.eye(2))
    assert dataset.axis_labels is None


def test_dataset_axis_labels():
    axis_labels = {'data': ('batch', 'features')}
    dataset = IterableDataset(numpy.eye(2), axis_labels=axis_labels)
    assert dataset.axis_labels == axis_labels


def test_sources_selection():
    features = [5, 6, 7, 1]
    targets = [1, 0, 1, 1]
    stream = DataStream(IterableDataset(OrderedDict(
        [('features', features), ('targets', targets)])))
    assert list(stream.get_epoch_iterator()) == list(zip(features, targets))

    stream = DataStream(IterableDataset(
        {'features': features, 'targets': targets},
        sources=('targets',)))
    assert list(stream.get_epoch_iterator()) == list(zip(targets))


def test_data_driven_epochs():
    class TestDataset(IterableDataset):
        sources = ('data',)

        def __init__(self):
            self.axis_labels = None
            self.data = [[1, 2, 3, 4],
                         [5, 6, 7, 8]]

        def open(self):
            epoch_iter = iter(self.data)
            data_iter = iter(next(epoch_iter))
            return (epoch_iter, data_iter)

        def next_epoch(self, state):
            try:
                data_iter = iter(next(state[0]))
                return (state[0], data_iter)
            except StopIteration:
                return self.open()

        def get_data(self, state, request):
            data = []
            for i in range(request):
                data.append(next(state[1]))
            return (data,)

    epochs = []
    epochs.append([([1],), ([2],), ([3],), ([4],)])
    epochs.append([([5],), ([6],), ([7],), ([8],)])
    stream = DataStream(TestDataset(), iteration_scheme=ConstantScheme(1))
    assert list(stream.get_epoch_iterator()) == epochs[0]
    assert list(stream.get_epoch_iterator()) == epochs[1]
    assert list(stream.get_epoch_iterator()) == epochs[0]

    stream.reset()
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]

    # test scheme resetting between epochs
    class TestScheme(BatchSizeScheme):

        def get_request_iterator(self):
            return iter([1, 2, 1, 3])

    epochs = []
    epochs.append([([1],), ([2, 3],), ([4],)])
    epochs.append([([5],), ([6, 7],), ([8],)])
    stream = DataStream(TestDataset(), iteration_scheme=TestScheme())
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]


def test_num_examples():
    assert_raises(ValueError, IterableDataset,
                  {'features': range(10), 'targets': range(7)})
    dataset = IterableDataset({'features': range(7),
                               'targets': range(7)})
    assert dataset.num_examples == 7
    dataset = IterableDataset(repeat(1))
    assert numpy.isnan(dataset.num_examples)
    x = numpy.random.rand(5, 3)
    y = numpy.random.rand(5, 4)
    dataset = IndexableDataset({'features': x, 'targets': y})
    assert dataset.num_examples == 5
    assert_raises(ValueError, IndexableDataset,
                  {'features': x, 'targets': y[:4]})
