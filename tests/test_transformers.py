import operator
from collections import OrderedDict

import numpy
from numpy.testing import assert_raises, assert_equal
from six.moves import zip, cPickle

from fuel import config
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.schemes import ConstantScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Transformer, Mapping, SortMapping, ForceFloatX, Filter, Cache, Batch,
    Padding, MultiProcessing, Unpack, Merge, SingleMapping, Flatten,
    ScaleAndShift, Cast, Rename, FilterSources)


class FlagDataStream(DataStream):
    close_called = False
    reset_called = False
    next_epoch_called = False

    def close(self):
        self.close_called = True
        super(FlagDataStream, self).close()

    def reset(self):
        self.reset_called = True
        super(FlagDataStream, self).reset()

    def next_epoch(self):
        self.next_epoch_called = True
        super(FlagDataStream, self).next_epoch()


class TestTransformer(object):
    def setUp(self):
        self.data_stream = FlagDataStream(IterableDataset([1, 2, 3]))
        self.transformer = Transformer(self.data_stream)

    def test_close(self):
        # Transformer.close should call its wrapped stream's close method
        self.transformer.close()
        assert self.data_stream.close_called

    def test_reset(self):
        # Transformer.reset should call its wrapped stream's reset method
        self.transformer.reset()
        assert self.data_stream.reset_called

    def test_next_epoch(self):
        # Transformer.next_epoch should call its wrapped stream's next_epoch
        # method
        self.transformer.next_epoch()
        assert self.data_stream.next_epoch_called

    def test_get_data_from_example_not_implemented(self):
        self.transformer.batch_input = False
        assert_raises(NotImplementedError, self.transformer.get_data)

    def test_get_data_from_batch_not_implemented(self):
        self.transformer.batch_input = True
        assert_raises(NotImplementedError, self.transformer.get_data)


class TestMapping(object):
    def setUp(self):
        self.data = [[1, 2, 3], [2, 3, 1], [3, 2, 1]]
        self.data_x = [[1, 2, 3], [2, 3, 1], [3, 2, 1]]
        self.data_y = [[6, 5, 4], [6, 5, 4], [6, 5, 4]]

    def test_mapping(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, lambda d: ([2 * i for i in d[0]],))
        assert_equal(list(transformer.get_epoch_iterator()),
                     list(zip([[2, 4, 6], [4, 6, 2], [6, 4, 2]])))

    def test_value_error_on_request(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, lambda d: ([2 * i for i in d[0]],))
        assert_raises(ValueError, transformer.get_data, [0, 1])

    def test_add_sources(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, lambda d: ([2 * i for i in d[0]],),
                              add_sources=('doubled',))
        assert_equal(transformer.sources, ('data', 'doubled'))
        assert_equal(list(transformer.get_epoch_iterator()),
                     list(zip(self.data, [[2, 4, 6], [4, 6, 2], [6, 4, 2]])))

    def test_sort_mapping_trivial_key(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, SortMapping(operator.itemgetter(0)))
        assert_equal(list(transformer.get_epoch_iterator()),
                     list(zip([[1, 2, 3]] * 3)))

    def test_sort_mapping_alternate_key(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, SortMapping(lambda x: -x[0]))
        assert_equal(list(transformer.get_epoch_iterator()),
                     list(zip([[3, 2, 1]] * 3)))

    def test_sort_mapping_reverse(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(
            stream, SortMapping(operator.itemgetter(0), reverse=True))
        assert_equal(list(transformer.get_epoch_iterator()),
                     list(zip([[3, 2, 1]] * 3)))

    def test_mapping_sort_multisource_ndarrays(self):
        data = OrderedDict([('x', numpy.array(self.data_x)),
                            ('y', numpy.array(self.data_y))])
        data_sorted = [(numpy.array([1, 2, 3]), numpy.array([6, 5, 4])),
                       (numpy.array([1, 2, 3]), numpy.array([4, 6, 5])),
                       (numpy.array([1, 2, 3]), numpy.array([4, 5, 6]))]
        stream = DataStream(IterableDataset(data))
        transformer = Mapping(
            stream, mapping=SortMapping(operator.itemgetter(0)))
        assert_equal(list(transformer.get_epoch_iterator()),
                     data_sorted)

    def test_mapping_sort_multisource(self):
        data = OrderedDict([('x', self.data_x), ('y', self.data_y)])
        data_sorted = [([1, 2, 3], [6, 5, 4]),
                       ([1, 2, 3], [4, 6, 5]),
                       ([1, 2, 3], [4, 5, 6])]
        stream = DataStream(IterableDataset(data))
        transformer = Mapping(
            stream, mapping=SortMapping(operator.itemgetter(0)))
        assert_equal(list(transformer.get_epoch_iterator()),
                     data_sorted)


def test_single_mapping_value_error_on_request():
    class IdentitySingleMapping(SingleMapping):
        def mapping(self, source):
            return source

    data_stream = DataStream(IndexableDataset([0, 1, 2]))
    transformer = IdentitySingleMapping(data_stream)
    assert_raises(ValueError, transformer.get_data, [0, 1])


def test_flatten():
    stream = DataStream(
        IndexableDataset(
            OrderedDict([('features', numpy.ones((4, 2, 2))),
                         ('targets', numpy.array([0, 1, 0, 1]))])),
        iteration_scheme=SequentialScheme(4, 2))
    wrapper = Flatten(stream, which_sources=('features',))
    assert_equal(
        list(wrapper.get_epoch_iterator()),
        [(numpy.ones((2, 4)), numpy.array([0, 1])),
         (numpy.ones((2, 4)), numpy.array([0, 1]))])


def test_scale_and_shift():
    stream = DataStream(
        IterableDataset(
            OrderedDict([('features', [1, 2, 3]), ('targets', [0, 1, 0])])))
    wrapper = ScaleAndShift(stream, 2, -1, which_sources=('targets',))
    assert list(wrapper.get_epoch_iterator()) == [(1, -1), (2, 1), (3, -1)]


def test_cast():
    stream = DataStream(
        IterableDataset(
            OrderedDict([
                ('features', numpy.array([1, 2, 3]).astype('float64')),
                ('targets', [0, 1, 0])])))
    wrapper = Cast(stream, 'float32', which_sources=('features',))
    assert_equal(
        list(wrapper.get_epoch_iterator()),
        [(numpy.array(1), 0), (numpy.array(2), 1), (numpy.array(3), 0)])
    assert all(f.dtype == 'float32' for f, t in wrapper.get_epoch_iterator())


class TestForceFloatX(object):
    def setUp(self):
        self.original_floatX = config.floatX
        config.floatX = 'float32'
        x = numpy.arange(6, dtype='float64').reshape((3, 2))
        y = numpy.arange(3, dtype='int64').reshape((3, 1))
        self.dataset = IterableDataset(OrderedDict([('x', x), ('y', y)]))

    def tearDown(self):
        config.floatX = self.original_floatX

    def test_force_floatx(self):
        transformer = ForceFloatX(DataStream(self.dataset))
        data = next(transformer.get_epoch_iterator())
        assert_equal(str(data[0].dtype), config.floatX)
        assert_equal(str(data[1].dtype), 'int64')

    def test_value_error_on_request(self):
        transformer = ForceFloatX(DataStream(self.dataset))
        assert_raises(ValueError, transformer.get_data, [0, 1])

    def test_axis_labels(self):
        axis_labels = {'x': ('batch', 'feature'), 'y': ('batch', 'index')}
        self.dataset.axis_labels = axis_labels
        transformer = ForceFloatX(DataStream(self.dataset))
        assert_equal(transformer.axis_labels, axis_labels)


def test_filter():
    data = [1, 2, 3]
    data_filtered = [1, 3]
    stream = DataStream(IterableDataset(data))
    wrapper = Filter(stream, lambda d: d[0] % 2 == 1)
    assert list(wrapper.get_epoch_iterator()) == list(zip(data_filtered))


class TestCache(object):
    def setUp(self):
        self.stream = Batch(
            DataStream(IterableDataset(range(100))), ConstantScheme(11))

    def test_cache_fills_correctly(self):
        cached_stream = Cache(self.stream, ConstantScheme(7))
        epoch = cached_stream.get_epoch_iterator()
        for (features,), cache_size in zip(epoch, [4, 8, 1, 5, 9, 2,
                                                   6, 10, 3, 7, 0, 4]):
            assert_equal(len(cached_stream.cache[0]), cache_size)

    def test_epoch_finishes_correctly(self):
        cached_stream = Cache(self.stream, ConstantScheme(7))
        data = list(cached_stream.get_epoch_iterator())
        assert_equal(len(data[-1][0]), 100 % 7)
        assert not cached_stream.cache[0]

    def test_epoch_transition(self):
        cached_stream = Cache(self.stream, ConstantScheme(7, times=3))
        for _, epoch in zip(range(2), cached_stream.iterate_epochs()):
            cache_sizes = [4, 8, 1]
            for i, (features,) in enumerate(epoch):
                assert_equal(len(cached_stream.cache[0]), cache_sizes[i])
                assert_equal(len(features), 7)
                assert_equal(list(range(100))[i * 7:(i + 1) * 7], features)
            assert_equal(i, 2)


class TestBatch(object):
    def test_strictness_0(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4, 5]))
        transformer = Batch(stream, ConstantScheme(2), strictness=0)
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.array([1, 2]),), (numpy.array([3, 4]),),
                      (numpy.array([5]),)])

    def test_strictness_1(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4, 5]))
        transformer = Batch(stream, ConstantScheme(2), strictness=1)
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.array([1, 2]),), (numpy.array([3, 4]),)])

    def test_strictness_2(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4, 5, 6]))
        transformer = Batch(stream, ConstantScheme(2), strictness=2)
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.array([1, 2]),), (numpy.array([3, 4]),),
                      (numpy.array([5, 6]),)])

    def test_strictness_2_error(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4, 5]))
        transformer = Batch(stream, ConstantScheme(2), strictness=2)
        assert_raises(ValueError, list, transformer.get_epoch_iterator())

    def test_value_error_on_request_none(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4, 5]))
        transformer = Batch(stream, ConstantScheme(2))
        assert_raises(ValueError, transformer.get_data, None)


class TestUnpack(object):
    def setUp(self):
        data = range(10)
        self.stream = Batch(
            DataStream(IterableDataset(data)),
            iteration_scheme=ConstantScheme(2))
        data_np = numpy.arange(10)
        self.stream_np = Batch(
            DataStream(IterableDataset(data_np)),
            iteration_scheme=ConstantScheme(2))

    def test_unpack(self):
        wrapper = Unpack(self.stream)
        epoch = wrapper.get_epoch_iterator()
        for i, v in enumerate(epoch):
            assert numpy.shape(v)[0] == 1
            assert v[0] == i

    def test_unpack_picklable(self):
        wrapper = Unpack(self.stream_np)
        epoch = wrapper.get_epoch_iterator()
        cPickle.dumps(epoch)


class TestPadding(object):
    def test_1d_sequences(self):
        stream = Batch(
            DataStream(
                IterableDataset([[1], [2, 3], [], [4, 5, 6], [7]])),
            ConstantScheme(2))
        transformer = Padding(stream)
        assert_equal(transformer.sources, ("data", "data_mask"))
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.array([[1, 0], [2, 3]]),
                       numpy.array([[1, 0], [1, 1]])),
                      (numpy.array([[0, 0, 0], [4, 5, 6]]),
                       numpy.array([[0, 0, 0], [1, 1, 1]])),
                      (numpy.array([[7]]), numpy.array([[1]]))])

    def test_2d_sequences(self):
        stream = Batch(
            DataStream(
                IterableDataset([numpy.ones((3, 4)), 2 * numpy.ones((2, 4))])),
            ConstantScheme(2))
        it = Padding(stream).get_epoch_iterator()
        data, mask = next(it)
        assert data.shape == (2, 3, 4)
        assert (data[0, :, :] == 1).all()
        assert (data[1, :2, :] == 2).all()
        assert (mask == numpy.array([[1, 1, 1], [1, 1, 0]])).all()

    def test_2d_sequences_error_on_unequal_shapes(self):
        stream = Batch(
            DataStream(
                IterableDataset([numpy.ones((3, 4)), 2 * numpy.ones((2, 3))])),
            ConstantScheme(2))
        assert_raises(ValueError, next, Padding(stream).get_epoch_iterator())

    def test_two_sources(self):
        transformer = Padding(Batch(
            DataStream(
                IterableDataset(
                    dict(features=[[1], [2, 3]], targets=[[4, 5, 6], [7]]))),
            ConstantScheme(2)))
        assert len(next(transformer.get_epoch_iterator())) == 4

    def test_mask_dtype(self):
        transformer = Padding(Batch(
            DataStream(
                IterableDataset(
                    dict(features=[[1], [2, 3]], targets=[[4, 5, 6], [7]]))),
            ConstantScheme(2)),
            mask_dtype='uint8')
        assert_equal(
            str(next(transformer.get_epoch_iterator())[1].dtype), 'uint8')

    def test_mask_sources(self):
        transformer = Padding(Batch(
            DataStream(
                IterableDataset(
                    OrderedDict([('features', [[1], [2, 3]]),
                                 ('targets', [[4, 5, 6], [7]])]))),
            ConstantScheme(2)),
            mask_sources=('features',))
        assert_equal(len(next(transformer.get_epoch_iterator())), 3)

    def test_value_error_on_request(self):
        transformer = Padding(Batch(
            DataStream(
                IterableDataset(
                    dict(features=[[1], [2, 3]], targets=[[4, 5, 6], [7]]))),
            ConstantScheme(2)))
        assert_raises(ValueError, transformer.get_data, [0, 1])


class TestMerge(object):
    def setUp(self):
        self.streams = (
            IterableDataset(['Hello world!']).get_example_stream(),
            IterableDataset(['Bonjour le monde!']).get_example_stream())

    def test_sources(self):
        transformer = Merge(self.streams, ('english', 'french'))
        assert_equal(transformer.sources, ('english', 'french'))

    def test_merge(self):
        transformer = Merge(self.streams, ('english', 'french'))
        assert_equal(next(transformer.get_epoch_iterator()),
                     ('Hello world!', 'Bonjour le monde!'))

    def test_as_dict(self):
        transformer = Merge(self.streams, ('english', 'french'))
        assert_equal(
            next(transformer.get_epoch_iterator(as_dict=True)),
            ({'english': 'Hello world!', 'french': 'Bonjour le monde!'}))

    def test_error_on_wrong_number_of_sources(self):
        assert_raises(ValueError, Merge, self.streams, ('english',))


class TestMultiprocessing(object):
    def setUp(self):
        stream = IterableDataset(range(100)).get_example_stream()
        self.transformer = Mapping(stream, lambda x: (x[0] + 1,))

    def test_multiprocessing(self):
        background = MultiProcessing(self.transformer)
        assert_equal(list(background.get_epoch_iterator()),
                     list(zip(range(1, 101))))

    def test_value_error_on_request(self):
        background = MultiProcessing(self.transformer)
        assert_raises(ValueError, background.get_data, [0, 1])


def test_rename():
    stream = DataStream(
        IndexableDataset(
            OrderedDict([('X', numpy.ones((4, 2, 2))),
                         ('y', numpy.array([0, 1, 0, 1]))])),
        iteration_scheme=SequentialScheme(4, 2))
    transformer = Rename(stream, {'X': 'features', 'y': 'targets'})
    assert_equal(transformer.sources, ('features', 'targets'))
    assert_equal(
        list(transformer.get_epoch_iterator()),
        [(numpy.ones((2, 2, 2)), numpy.array([0, 1])),
         (numpy.ones((2, 2, 2)), numpy.array([0, 1]))])
    assert_raises(ValueError, transformer.get_data, [0, 1])
    assert_raises(KeyError, Rename, stream, {'Z': 'features'})


def test_filter_sources():
    stream = DataStream(
        IndexableDataset(
            OrderedDict([('features', numpy.ones((4, 2, 2))),
                         ('targets', numpy.array([0, 1, 0, 1]))])),
        iteration_scheme=SequentialScheme(4, 2))

    transformer = FilterSources(stream, sources=("features",))

    assert_equal(transformer.sources, ('features',))
    assert len(next(transformer.get_epoch_iterator())) == 1

    transformer = FilterSources(stream, sources=("features", "targets"))

    assert_equal(transformer.sources, ('features', 'targets'))
    assert len(next(transformer.get_epoch_iterator())) == 2

    transformer = FilterSources(stream, sources=("targets", "features"))

    assert_equal(transformer.sources, ('features', 'targets'))
    assert len(next(transformer.get_epoch_iterator())) == 2

    assert_raises(ValueError, transformer.get_data, [0, 1])
    assert_raises(ValueError, FilterSources, stream, ['error', 'targets'])
