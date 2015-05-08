import operator
from collections import OrderedDict

import numpy
from numpy.testing import assert_raises, assert_equal
from six.moves import zip

from fuel import config
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.schemes import ConstantScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Transformer, Mapping, SortMapping, ForceFloatX, Filter, Cache, Batch,
    Padding, MultiProcessing, Unpack, Merge, Flatten, ScaleAndShift, Cast)


class IdentityTransformer(Transformer):
    def get_data(self, request=None):
        return self.data_stream.get_data(request)


def test_mapping():
    data = [1, 2, 3]
    data_doubled = [2, 4, 6]
    stream = DataStream(IterableDataset(data))
    wrapper1 = Mapping(stream, lambda d: (2 * d[0],))
    assert list(wrapper1.get_epoch_iterator()) == list(zip(data_doubled))
    wrapper2 = Mapping(stream, lambda d: (2 * d[0],), add_sources=("doubled",))
    assert wrapper2.sources == ("data", "doubled")
    assert list(wrapper2.get_epoch_iterator()) == list(zip(data, data_doubled))


def test_mapping_sort():
    data = [[1, 2, 3],
            [2, 3, 1],
            [3, 2, 1]]
    data_sorted = [[1, 2, 3]] * 3
    data_sorted_rev = [[3, 2, 1]] * 3
    stream = DataStream(IterableDataset(data))
    wrapper1 = Mapping(stream, SortMapping(operator.itemgetter(0)))
    assert list(wrapper1.get_epoch_iterator()) == list(zip(data_sorted))
    wrapper2 = Mapping(stream, SortMapping(lambda x: -x[0]))
    assert list(wrapper2.get_epoch_iterator()) == list(zip(data_sorted_rev))
    wrapper3 = Mapping(
        stream, SortMapping(operator.itemgetter(0), reverse=True))
    assert list(wrapper3.get_epoch_iterator()) == list(zip(data_sorted_rev))


def test_mapping_sort_multisource_ndarrays():
    data = OrderedDict()
    data['x'] = [numpy.array([1, 2, 3]),
                 numpy.array([2, 3, 1]),
                 numpy.array([3, 2, 1])]
    data['y'] = [numpy.array([6, 5, 4]),
                 numpy.array([6, 5, 4]),
                 numpy.array([6, 5, 4])]
    data_sorted = [(numpy.array([1, 2, 3]), numpy.array([6, 5, 4])),
                   (numpy.array([1, 2, 3]), numpy.array([4, 6, 5])),
                   (numpy.array([1, 2, 3]), numpy.array([4, 5, 6]))]
    stream = DataStream(IterableDataset(data))
    wrapper = Mapping(stream, mapping=SortMapping(operator.itemgetter(0)))
    for output, ground_truth in zip(wrapper.get_epoch_iterator(), data_sorted):
        assert len(output) == len(ground_truth)
        assert (output[0] == ground_truth[0]).all()
        assert (output[1] == ground_truth[1]).all()


def test_mapping_sort_multisource():
    data = OrderedDict()
    data['x'] = [[1, 2, 3], [2, 3, 1], [3, 2, 1]]
    data['y'] = [[6, 5, 4], [6, 5, 4], [6, 5, 4]]
    data_sorted = [([1, 2, 3], [6, 5, 4]),
                   ([1, 2, 3], [4, 6, 5]),
                   ([1, 2, 3], [4, 5, 6])]
    stream = DataStream(IterableDataset(data))
    wrapper = Mapping(stream, mapping=SortMapping(operator.itemgetter(0)))
    assert list(wrapper.get_epoch_iterator()) == data_sorted


def test_flatten():
    stream = DataStream(
        IndexableDataset({'features': numpy.ones((4, 2, 2)),
                         'targets': numpy.array([0, 1, 0, 1])}),
        iteration_scheme=SequentialScheme(4, 2))
    wrapper = Flatten(stream, which_sources=('features',))
    assert_equal(
        list(wrapper.get_epoch_iterator()),
        [(numpy.ones((2, 4)), numpy.array([0, 1])),
         (numpy.ones((2, 4)), numpy.array([0, 1]))])


def test_scale_and_shift():
    stream = DataStream(
        IterableDataset({'features': [1, 2, 3], 'targets': [0, 1, 0]}))
    wrapper = ScaleAndShift(stream, 2, -1, which_sources=('targets',))
    assert list(wrapper.get_epoch_iterator()) == [(1, -1), (2, 1), (3, -1)]


def test_cast():
    stream = DataStream(
        IterableDataset({'features': numpy.array([1, 2, 3]).astype('float64'),
                         'targets': [0, 1, 0]}))
    wrapper = Cast(stream, 'float32', which_sources=('features',))
    assert_equal(
        list(wrapper.get_epoch_iterator()),
        [(numpy.array(1), 0), (numpy.array(2), 1), (numpy.array(3), 0)])
    assert all(f.dtype == 'float32' for f, t in wrapper.get_epoch_iterator())


def test_force_floatx():
    x = [numpy.array(d, dtype="float64") for d in [[1, 2], [3, 4], [5, 6]]]
    y = [numpy.array(d, dtype="int64") for d in [1, 2, 3]]
    dataset = IterableDataset(OrderedDict([("x", x), ("y", y)]))
    wrapper = ForceFloatX(DataStream(dataset))
    data = next(wrapper.get_epoch_iterator())
    assert str(data[0].dtype) == config.floatX
    assert str(data[1].dtype) == "int64"


def test_force_floatx_axis_labels():
    x = numpy.eye(2).astype('float64')
    axis_labels = {'x': ('batch', 'feature')}
    dataset = IterableDataset({'x': x}, axis_labels=axis_labels)
    stream = ForceFloatX(DataStream(dataset))
    assert stream.axis_labels == axis_labels


def test_filter():
    data = [1, 2, 3]
    data_filtered = [1, 3]
    stream = DataStream(IterableDataset(data))
    wrapper = Filter(stream, lambda d: d[0] % 2 == 1)
    assert list(wrapper.get_epoch_iterator()) == list(zip(data_filtered))


def test_cache():
    dataset = IterableDataset(range(100))
    stream = DataStream(dataset)
    batched_stream = Batch(stream, ConstantScheme(11))
    cached_stream = Cache(batched_stream, ConstantScheme(7))
    epoch = cached_stream.get_epoch_iterator()

    # Make sure that cache is filled as expected
    for (features,), cache_size in zip(epoch, [4, 8, 1, 5, 9, 2,
                                               6, 10, 3, 7, 0, 4]):
        assert len(cached_stream.cache[0]) == cache_size

    # Make sure that the epoch finishes correctly
    for (features,) in cached_stream.get_epoch_iterator():
        pass
    assert len(features) == 100 % 7
    assert not cached_stream.cache[0]

    # Ensure that the epoch transition is correct
    cached_stream = Cache(batched_stream, ConstantScheme(7, times=3))
    for _, epoch in zip(range(2), cached_stream.iterate_epochs()):
        cache_sizes = [4, 8, 1]
        for i, (features,) in enumerate(epoch):
            assert len(cached_stream.cache[0]) == cache_sizes[i]
            assert len(features) == 7
            assert numpy.all(list(range(100))[i * 7:(i + 1) * 7] == features)
        assert i == 2


def test_batch():
    stream = DataStream(IterableDataset([1, 2, 3, 4, 5]))
    wrapper = Batch(stream, ConstantScheme(2))
    batches = list(wrapper.get_epoch_iterator())
    expected = [(numpy.array([1, 2]),),
                (numpy.array([3, 4]),),
                (numpy.array([5]),)]
    assert len(batches) == len(expected)
    for b, e in zip(batches, expected):
        assert (b[0] == e[0]).all()

    # Check the `strict` flag
    def try_strict(strictness):
        return list(Batch(stream, ConstantScheme(2), strictness=strictness)
                    .get_epoch_iterator())
    assert_raises(ValueError, try_strict, 2)
    assert len(try_strict(1)) == 2
    stream2 = DataStream(IterableDataset([1, 2, 3, 4, 5, 6]))
    assert len(list(Batch(stream2, ConstantScheme(2), strictness=2)
                    .get_epoch_iterator())) == 3


def test_unpack():
    data = range(10)
    stream = Batch(
        DataStream(IterableDataset(data)), iteration_scheme=ConstantScheme(2))
    wrapper = Unpack(stream)
    epoch = wrapper.get_epoch_iterator()
    for i, v in enumerate(epoch):
        assert numpy.shape(v)[0] == 1
        assert v[0] == i


def test_padding():
    # 1-D sequences
    stream = Batch(
        DataStream(
            IterableDataset([[1], [2, 3], [], [4, 5, 6], [7]])),
        ConstantScheme(2))
    mask_stream = Padding(stream)
    assert mask_stream.sources == ("data", "data_mask")
    it = mask_stream.get_epoch_iterator()
    data, mask = next(it)
    assert (data == numpy.array([[1, 0], [2, 3]])).all()
    assert (mask == numpy.array([[1, 0], [1, 1]])).all()
    data, mask = next(it)
    assert (data == numpy.array([[0, 0, 0], [4, 5, 6]])).all()
    assert (mask == numpy.array([[0, 0, 0], [1, 1, 1]])).all()
    data, mask = next(it)
    assert (data == numpy.array([[7]])).all()
    assert (mask == numpy.array([[1]])).all()

    # 2D sequences
    stream2 = Batch(
        DataStream(
            IterableDataset([numpy.ones((3, 4)), 2 * numpy.ones((2, 4))])),
        ConstantScheme(2))
    it = Padding(stream2).get_epoch_iterator()
    data, mask = next(it)
    assert data.shape == (2, 3, 4)
    assert (data[0, :, :] == 1).all()
    assert (data[1, :2, :] == 2).all()
    assert (mask == numpy.array([[1, 1, 1], [1, 1, 0]])).all()

    # 2 sources
    stream3 = Padding(Batch(
        DataStream(
            IterableDataset(
                dict(features=[[1], [2, 3]], targets=[[4, 5, 6], [7]]))),
        ConstantScheme(2)))
    assert len(next(stream3.get_epoch_iterator())) == 4


def test_merge():
    english = IterableDataset(['Hello world!'])
    french = IterableDataset(['Bonjour le monde!'])
    streams = (english.get_example_stream(),
               french.get_example_stream())
    merged_stream = Merge(streams, ('english', 'french'))
    assert merged_stream.sources == ('english', 'french')
    assert (next(merged_stream.get_epoch_iterator()) ==
            ('Hello world!', 'Bonjour le monde!'))


def test_multiprocessing():
    stream = IterableDataset(range(100)).get_example_stream()
    plus_one = Mapping(stream, lambda x: (x[0] + 1,))
    background = MultiProcessing(plus_one)
    for a, b in zip(background.get_epoch_iterator(), range(1, 101)):
        assert a == (b,)
