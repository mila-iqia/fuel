from collections import OrderedDict

import numpy
from numpy.testing import assert_raises, assert_equal
from picklable_itertools import repeat
from six.moves import zip, range, cPickle

from fuel.datasets import Dataset, IterableDataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, BatchSizeScheme, ConstantScheme
from fuel.transformers import Mapping


class TestDataset(object):
    def setUp(self):
        self.data = [1, 2, 3]
        self.stream = DataStream(IterableDataset(self.data))

    def test_one_example_at_a_time(self):
        assert_equal(
            list(self.stream.get_epoch_iterator()), list(zip(self.data)))

    def test_multiple_epochs(self):
        for i, epoch in zip(range(2), self.stream.iterate_epochs()):
            assert list(epoch) == list(zip(self.data))

    def test_as_dict(self):
        assert_equal(
            next(self.stream.get_epoch_iterator(as_dict=True)), {"data": 1})

    def test_value_error_on_no_provided_sources(self):
        class FaultyDataset(Dataset):
            def get_data(self, state=None, request=None):
                pass
        assert_raises(ValueError, FaultyDataset, self.data)

    def test_value_error_on_nonexistent_sources(self):
        def instantiate_dataset():
            return IterableDataset(self.data, sources=('dummy',))
        assert_raises(ValueError, instantiate_dataset)

    def test_default_transformer(self):
        class DoublingDataset(IterableDataset):
            def apply_default_transformer(self, stream):
                return Mapping(
                    stream, lambda sources: tuple(2 * s for s in sources))
        dataset = DoublingDataset(self.data)
        stream = dataset.apply_default_transformer(DataStream(dataset))
        assert_equal(list(stream.get_epoch_iterator()), [(2,), (4,), (6,)])

    def test_no_axis_labels(self):
        assert IterableDataset(self.data).axis_labels is None

    def test_axis_labels(self):
        axis_labels = {'data': ('batch',)}
        dataset = IterableDataset(self.data, axis_labels=axis_labels)
        assert dataset.axis_labels == axis_labels

    def test_attribute_error_on_no_example_iteration_scheme(self):
        class FaultyDataset(Dataset):
            provides_sources = ('data',)

            def get_data(self, state=None, request=None):
                pass

        def get_example_iteration_scheme():
            return FaultyDataset().example_iteration_scheme

        assert_raises(AttributeError, get_example_iteration_scheme)

    def test_example_iteration_scheme(self):
        scheme = ConstantScheme(2)

        class MinimalDataset(Dataset):
            provides_sources = ('data',)
            _example_iteration_scheme = scheme

            def get_data(self, state=None, request=None):
                pass

        assert MinimalDataset().example_iteration_scheme is scheme

    def test_filter_sources(self):
        dataset = IterableDataset(
            OrderedDict([('1', [1, 2]), ('2', [3, 4])]), sources=('1',))
        assert_equal(dataset.filter_sources(([1, 2], [3, 4])), ([1, 2],))


class TestIterableDataset(object):
    def test_value_error_on_non_iterable_dict(self):
        assert_raises(ValueError, IterableDataset, {'x': None, 'y': None})

    def test_value_error_on_non_iterable(self):
        assert_raises(ValueError, IterableDataset, None)

    def test_value_error_get_data_none_state(self):
        assert_raises(
            ValueError, IterableDataset([1, 2, 3]).get_data, None, None)

    def test_value_error_get_data_request(self):
        assert_raises(
            ValueError, IterableDataset([1, 2, 3]).get_data, [1, 2, 3], True)


class TestIndexableDataset(object):
    def test_getattr(self):
        assert_equal(getattr(IndexableDataset({'a': (1, 2)}), 'a'), (1, 2))

    def test_value_error_on_non_iterable(self):
        assert_raises(ValueError, IterableDataset, None)

    def test_value_error_get_data_state(self):
        assert_raises(
            ValueError, IndexableDataset([1, 2, 3]).get_data, True, [1, 2])

    def test_value_error_get_data_none_request(self):
        assert_raises(
            ValueError, IndexableDataset([1, 2, 3]).get_data, None, None)

    def test_pickling(self):
        cPickle.loads(cPickle.dumps(IndexableDataset({'a': (1, 2)})))

    def test_batch_iteration_scheme_with_lists(self):
        """Batch schemes should work with more than ndarrays."""
        data = IndexableDataset(OrderedDict([('foo', list(range(50))),
                                             ('bar', list(range(1, 51)))]))
        stream = DataStream(data,
                            iteration_scheme=ShuffledScheme(data.num_examples,
                                                            5))
        returned = [sum(batches, []) for batches in
                    zip(*list(stream.get_epoch_iterator()))]
        assert set(returned[0]) == set(range(50))
        assert set(returned[1]) == set(range(1, 51))


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
