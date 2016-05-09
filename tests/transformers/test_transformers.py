import logging
import operator
from collections import OrderedDict

import numpy
from numpy.testing import assert_raises, assert_equal
from six.moves import zip, cPickle

from fuel import config
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.schemes import (ConstantScheme, SequentialScheme,
                          SequentialExampleScheme)
from fuel.streams import DataStream
from fuel.transformers import (
    ExpectsAxisLabels, Transformer, Mapping, SortMapping, ForceFloatX, Filter,
    Cache, Batch, Padding, MultiProcessing, Unpack, Merge,
    SourcewiseTransformer, Flatten, ScaleAndShift, Cast, Rename, FilterSources)
from fuel.transformers.defaults import ToBytes


class FlagDataStream(DataStream):
    close_called = False
    reset_called = False
    next_epoch_called = False
    produces_examples = True

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
        self.transformer = Transformer(
            self.data_stream, produces_examples=True)

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

    def test_value_error_on_request(self):
        assert_raises(ValueError, self.transformer.get_data, [0, 1])

    def test_batch_to_example_and_vice_versa_not_implemented(self):
        self.transformer.produces_examples = False
        self.transformer.get_epoch_iterator()
        assert_raises(NotImplementedError, self.transformer.get_data)

    def test_transform_example_not_implemented_by_default(self):
        assert_raises(
            NotImplementedError, self.transformer.transform_example, None)

    def test_transform_batch_not_implemented_by_default(self):
        assert_raises(
            NotImplementedError, self.transformer.transform_batch, None)


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

    def test_value_error_on_request(self):
        stream = DataStream(IterableDataset(self.data))
        transformer = Mapping(stream, lambda d: ([2 * i for i in d[0]],))
        assert_raises(ValueError, transformer.get_data, [0, 1])


class TestSourcewiseTransformer(object):
    def test_transform_source_example_not_implemented(self):
        transformer = SourcewiseTransformer(
            DataStream(IterableDataset([1, 2])), True)
        assert_raises(
            NotImplementedError, transformer.transform_source_example,
            None, 'foo')

    def test_transform_source_batch_not_implemented(self):
        transformer = SourcewiseTransformer(
            DataStream(IterableDataset([1, 2])), True)
        assert_raises(
            NotImplementedError, transformer.transform_source_batch,
            None, 'foo')


class TestFlatten(object):
    def setUp(self):
        self.data = OrderedDict(
            [('features', numpy.ones((4, 2, 2))),
             ('targets', numpy.array([[0], [1], [0], [1]]))])

    def test_flatten_batches(self):
        wrapper = Flatten(
            DataStream(IndexableDataset(self.data),
                       iteration_scheme=SequentialScheme(4, 2)),
            which_sources=('features',))
        assert_equal(
            list(wrapper.get_epoch_iterator()),
            [(numpy.ones((2, 4)), numpy.array([[0], [1]])),
             (numpy.ones((2, 4)), numpy.array([[0], [1]]))])

    def test_axis_labels_on_flatten_batches(self):
        wrapper = Flatten(
            DataStream(IndexableDataset(self.data),
                       iteration_scheme=SequentialScheme(4, 2),
                       axis_labels={'features': ('batch', 'width', 'height'),
                                    'targets': ('batch', 'index')}),
            which_sources=('features',))
        assert_equal(wrapper.axis_labels, {'features': ('batch', 'feature'),
                                           'targets': ('batch', 'index')})

    def test_axis_labels_on_flatten_batches_with_none(self):
        wrapper = Flatten(
            DataStream(IndexableDataset(self.data),
                       iteration_scheme=SequentialScheme(4, 2),
                       axis_labels={'features': None,
                                    'targets': ('batch', 'index')}),
            which_sources=('features',))
        assert_equal(wrapper.axis_labels, {'features': None,
                                           'targets': ('batch', 'index')})

    def test_flatten_examples(self):
        wrapper = Flatten(
            DataStream(IndexableDataset(self.data),
                       iteration_scheme=SequentialExampleScheme(4)),
            which_sources=('features',))
        assert_equal(
            list(wrapper.get_epoch_iterator()),
            [(numpy.ones(4), 0), (numpy.ones(4), 1)] * 2)

    def test_axis_labels_on_flatten_examples(self):
        wrapper = Flatten(
            DataStream(IndexableDataset(self.data),
                       iteration_scheme=SequentialExampleScheme(4),
                       axis_labels={'features': ('batch', 'width', 'height'),
                                    'targets': ('batch', 'index')}),
            which_sources=('features',))
        assert_equal(wrapper.axis_labels, {'features': ('feature',),
                                           'targets': ('index',)})


class TestScaleAndShift(object):
    def setUp(self):
        dataset = IterableDataset(
            OrderedDict([('features', [1, 2, 3]), ('targets', [0, 1, 0])]),
            axis_labels={'features': ('batch'), 'targets': ('batch')})
        self.stream = DataStream(dataset)
        self.wrapper = ScaleAndShift(
            self.stream, 2, -1, which_sources=('targets',))

    def test_scale_and_shift(self):
        assert_equal(list(self.wrapper.get_epoch_iterator()),
                     [(1, -1), (2, 1), (3, -1)])

    def test_axis_labels_are_passed_through(self):
        assert_equal(self.wrapper.axis_labels, self.stream.axis_labels)


class TestCast(object):
    def setUp(self):
        dataset = IterableDataset(
            OrderedDict([
                ('features', numpy.array([1, 2, 3]).astype('float64')),
                ('targets', [0, 1, 0])]),
            axis_labels={'features': ('batch'), 'targets': ('batch')})
        self.stream = DataStream(dataset)
        self.wrapper = Cast(
            self.stream, 'float32', which_sources=('features',))

    def test_cast(self):
        assert_equal(
            list(self.wrapper.get_epoch_iterator()),
            [(numpy.array(1), 0), (numpy.array(2), 1), (numpy.array(3), 0)])
        assert all(
            f.dtype == 'float32' for f, t in self.wrapper.get_epoch_iterator())

    def test_axis_labels_are_passed_through(self):
        assert_equal(self.wrapper.axis_labels, self.stream.axis_labels)


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

    def test_axis_labels_are_passed_through(self):
        axis_labels = {'x': ('batch', 'feature'), 'y': ('batch', 'index')}
        self.dataset.axis_labels = axis_labels
        stream = DataStream(self.dataset)
        transformer = ForceFloatX(stream)
        assert_equal(transformer.axis_labels, stream.axis_labels)


class TestFilter(object):
    def test_filter_examples(self):
        data = [1, 2, 3]
        data_filtered = [1, 3]
        stream = DataStream(IterableDataset(data))
        wrapper = Filter(stream, lambda d: d[0] % 2 == 1)
        assert_equal(list(wrapper.get_epoch_iterator()),
                     list(zip(data_filtered)))

    def test_filter_batches(self):
        data = [1, 2, 3, 4]
        data_filtered = [([3, 4],)]
        stream = DataStream(IndexableDataset(data),
                            iteration_scheme=SequentialScheme(4, 2))
        wrapper = Filter(stream, lambda d: d[0][0] % 3 == 0)
        assert_equal(list(wrapper.get_epoch_iterator()), data_filtered)

    def test_axis_labels_are_passed_through(self):
        stream = DataStream(
            IndexableDataset(
                {'features': [1, 2, 3, 4]},
                axis_labels={'features': ('batch',)}),
            iteration_scheme=SequentialScheme(4, 2))
        wrapper = Filter(stream, lambda d: d[0][0] % 3 == 0)
        assert_equal(wrapper.axis_labels, stream.axis_labels)


class TestCache(object):
    def setUp(self):
        self.stream = Batch(DataStream(IterableDataset(range(100))),
                            ConstantScheme(11))

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

        stream = Batch(DataStream(IterableDataset(range(3000))),
                       ConstantScheme(3200))

        cached_stream = Cache(stream, ConstantScheme(64))
        data = list(cached_stream.get_epoch_iterator())
        assert_equal(len(data[-1][0]), 3000 % 64)
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

    def test_value_error_on_non_batchsizescheme(self):
        assert_raises(ValueError, Cache, self.stream, SequentialScheme(4, 2))

    def test_value_error_on_none_request(self):
        cached_stream = Cache(self.stream, ConstantScheme(7))
        cached_stream.get_epoch_iterator()
        assert_raises(ValueError, cached_stream.get_data, None)

    def test_axis_labels_passed_on_by_default(self):
        self.stream.axis_labels = {'features': ('batch', 'index')}
        cached_stream = Cache(self.stream, ConstantScheme(7))
        assert_equal(cached_stream.axis_labels, self.stream.axis_labels)


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

    def test_adds_batch_to_axis_labels(self):
        stream = DataStream(
            IterableDataset(
                {'features': [1, 2, 3, 4, 5]},
                axis_labels={'features': ('index',)}))
        transformer = Batch(stream, ConstantScheme(2), strictness=0)
        assert_equal(transformer.axis_labels, {'features': ('batch', 'index')})

    def test_value_error_on_batch_stream(self):
        stream = DataStream(IndexableDataset([1, 2, 3, 4]),
                            iteration_scheme=SequentialScheme(4, 2))
        assert_raises(ValueError, Batch, stream, SequentialScheme(4, 2))

    def test_value_error_on_example_scheme(self):
        stream = DataStream(IterableDataset([1, 2, 3, 4]))
        assert_raises(ValueError, Batch, stream, SequentialExampleScheme(4))


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

    def test_value_error_on_example_stream(self):
        stream = DataStream(
            IterableDataset(
                dict(features=[[1], [2, 3]],
                     targets=[[4, 5, 6], [7]])))
        assert_raises(ValueError, Unpack, stream)

    def test_value_error_on_request(self):
        wrapper = Unpack(self.stream)
        assert_raises(ValueError, wrapper.get_data, [0, 1])

    def test_axis_labels_default(self):
        self.stream.axis_labels = {'features': ('batch', 'index')}
        wrapper = Unpack(self.stream)
        assert_equal(wrapper.axis_labels, {'features': ('index',)})


class TestPadding(object):
    def test_1d_sequences(self):
        stream = Batch(
            DataStream(IterableDataset([[1], [2, 3], [], [4, 5, 6], [7]])),
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
                    OrderedDict([
                        ('features', [[1], [2, 3]]),
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

    def test_value_error_on_example_stream(self):
        stream = DataStream(
            IterableDataset(
                dict(features=[[1], [2, 3]], targets=[[4, 5, 6], [7]])))
        assert_raises(ValueError, Padding, stream)


class TestMerge(object):
    def setUp(self):
        self.streams = (
            DataStream(IterableDataset(['Hello world!'])),
            DataStream(IterableDataset(['Bonjour le monde!'])))
        self.batch_streams = (
            Batch(DataStream(IterableDataset(['Hello world!', 'Hi!'])),
                  iteration_scheme=ConstantScheme(2)),
            Batch(DataStream(IterableDataset(['Bonjour le monde!', 'Salut!'])),
                  iteration_scheme=ConstantScheme(2)))
        self.transformer = Merge(
            self.streams, ('english', 'french'))
        self.batch_transformer = Merge(
            self.batch_streams, ('english', 'french'))

    def test_sources(self):
        assert_equal(self.transformer.sources, ('english', 'french'))

    def test_merge(self):
        it = self.transformer.get_epoch_iterator()
        assert_equal(next(it), ('Hello world!', 'Bonjour le monde!'))
        assert_raises(StopIteration, next, it)
        # There used to be problems with reseting Merge, for which
        # reason we regression-test it as follows:
        it = self.transformer.get_epoch_iterator()
        assert_equal(next(it), ('Hello world!', 'Bonjour le monde!'))
        assert_raises(StopIteration, next, it)

    def test_batch_merge(self):
        it = self.batch_transformer.get_epoch_iterator()
        assert_equal(next(it),
                     (('Hello world!', 'Hi!'),
                      ('Bonjour le monde!', 'Salut!')))
        assert_raises(StopIteration, next, it)
        # There used to be problems with reseting Merge, for which
        # reason we regression-test it as follows:
        it = self.batch_transformer.get_epoch_iterator()
        assert_equal(next(it),
                     (('Hello world!', 'Hi!'),
                      ('Bonjour le monde!', 'Salut!')))
        assert_raises(StopIteration, next, it)

    def test_merge_batch_streams(self):
        it = self.transformer.get_epoch_iterator()
        assert_equal(next(it), ('Hello world!', 'Bonjour le monde!'))
        assert_raises(StopIteration, next, it)
        # There used to be problems with reseting Merge, for which
        # reason we regression-test it as follows:
        it = self.transformer.get_epoch_iterator()
        assert_equal(next(it), ('Hello world!', 'Bonjour le monde!'))
        assert_raises(StopIteration, next, it)

    def test_as_dict(self):
        assert_equal(
            next(self.transformer.get_epoch_iterator(as_dict=True)),
            ({'english': 'Hello world!', 'french': 'Bonjour le monde!'}))

    def test_error_on_wrong_number_of_sources(self):
        assert_raises(ValueError, Merge, self.streams, ('english',))

    def test_value_error_on_different_stream_output_type(self):
        spanish_stream = DataStream(IndexableDataset(['Hola mundo!']),
                                    iteration_scheme=SequentialScheme(2, 2))
        assert_raises(ValueError, Merge, self.streams + (spanish_stream,),
                      ('english', 'french', 'spanish'))

    def test_close_calls_close_on_all_streams(self):
        streams = [FlagDataStream(IterableDataset([1, 2, 3])),
                   FlagDataStream(IterableDataset([4, 5, 6])),
                   FlagDataStream(IterableDataset([7, 8, 9]))]
        transformer = Merge(streams, ('1', '2', '3'))
        transformer.close()
        assert all(stream.close_called for stream in streams)

    def test_next_epoch_calls_next_epoch_on_all_streams(self):
        streams = [FlagDataStream(IterableDataset([1, 2, 3])),
                   FlagDataStream(IterableDataset([4, 5, 6])),
                   FlagDataStream(IterableDataset([7, 8, 9]))]
        transformer = Merge(streams, ('1', '2', '3'))
        transformer.next_epoch()
        assert all(stream.next_epoch_called for stream in streams)

    def test_reset_calls_reset_on_all_streams(self):
        streams = [FlagDataStream(IterableDataset([1, 2, 3])),
                   FlagDataStream(IterableDataset([4, 5, 6])),
                   FlagDataStream(IterableDataset([7, 8, 9]))]
        transformer = Merge(streams, ('1', '2', '3'))
        transformer.reset()
        assert all(stream.reset_called for stream in streams)


class TestMultiprocessing(object):
    def setUp(self):
        stream = DataStream(IterableDataset(range(100)))
        self.transformer = Mapping(stream, lambda x: (x[0] + 1,))

    def test_multiprocessing(self):
        background = MultiProcessing(self.transformer)
        assert_equal(list(background.get_epoch_iterator()),
                     list(zip(range(1, 101))))

    def test_value_error_on_request(self):
        background = MultiProcessing(self.transformer)
        assert_raises(ValueError, background.get_data, [0, 1])

    def test_axis_labels_passed_on_by_default(self):
        self.transformer.axis_labels = {'features': ('batch', 'index')}
        background = MultiProcessing(self.transformer)
        assert_equal(background.axis_labels, self.transformer.axis_labels)


class TestRename(object):
    def setUp(self):
        self.stream = DataStream(
            IndexableDataset(
                OrderedDict([('X', numpy.ones((4, 2, 2))),
                             ('y', numpy.array([0, 1, 0, 1]))]),
                axis_labels={'X': ('batch', 'width', 'height'),
                             'y': ('batch',)}),
            iteration_scheme=SequentialScheme(4, 2))
        self.transformer = Rename(
            self.stream, {'X': 'features', 'y': 'targets'})

    def test_renames_sources(self):
        assert_equal(self.transformer.sources, ('features', 'targets'))

    def test_leaves_data_unchanged(self):
        assert_equal(
            list(self.transformer.get_epoch_iterator()),
            [(numpy.ones((2, 2, 2)), numpy.array([0, 1])),
             (numpy.ones((2, 2, 2)), numpy.array([0, 1]))])

    def test_raises_error_on_nonexistent_source_name(self):
        assert_raises(KeyError, Rename, self.stream, {'Z': 'features'})

    def test_raises_on_invalid_kwargs(self):
        assert_raises(ValueError, Rename, self.stream,
                      {'X': 'features'}, on_non_existent='foo')

    def test_name_clash(self):
        assert_raises(KeyError, Rename, self.stream, {'X': 'y'})

    def test_not_really_a_name_clash(self):
        try:
            # This should not raise an error, because we're ignoring
            # non-existent sources. So renaming a non-existent source
            # cannot create a name clash.
            Rename(self.stream, {'foobar': 'y'}, on_non_existent='ignore')
        except KeyError:
            assert False   # Regression.

    def test_name_swap(self):
        assert_equal(Rename(self.stream,
                            {'X': 'y', 'y': 'X'},
                            on_non_existent='ignore').sources,
                     ('y', 'X'))

    def test_raises_on_not_one_to_one(self):
        assert_raises(KeyError, Rename, self.stream, {'X': 'features',
                                                      'y': 'features'})

    def test_intentionally_ignore_missing(self):
        assert_equal(Rename(self.stream,
                            {'X': 'features', 'y': 'targets',
                             'Z': 'fudgesicle'},
                            on_non_existent='ignore').sources,
                     ('features', 'targets'))

    def test_not_one_to_one_ok_if_not_a_source_in_data_stream(self):
        assert_equal(Rename(self.stream,
                            {'X': 'features', 'y': 'targets',
                             'Z': 'targets'},
                            on_non_existent='ignore').sources,
                     ('features', 'targets'))

    def test_renames_axis_labels(self):
        assert_equal(self.transformer.axis_labels,
                     {'features': ('batch', 'width', 'height'),
                      'targets': ('batch',)})


class TestFilterSources(object):
    def setUp(self):
        self.stream = DataStream(
            IndexableDataset(
                OrderedDict([('features', numpy.ones((4, 2, 2))),
                             ('targets', numpy.array([0, 1, 0, 1]))]),
                axis_labels={'features': ('batch', 'width', 'height'),
                             'targets': ('batch',)}),
            iteration_scheme=SequentialScheme(4, 2))

    def test_works_on_sourcessubset(self):
        transformer = FilterSources(self.stream, sources=("features",))
        assert_equal(transformer.sources, ('features',))
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.ones((2, 2, 2)),), (numpy.ones((2, 2, 2)),)])

    def test_works_on_all_sources(self):
        transformer = FilterSources(
            self.stream, sources=("features", "targets"))
        assert_equal(transformer.sources, ('features', 'targets'))
        assert_equal(list(transformer.get_epoch_iterator()),
                     [(numpy.ones((2, 2, 2)), numpy.array([0, 1])),
                      (numpy.ones((2, 2, 2)), numpy.array([0, 1]))])

    def test_works_on_unsorted_sources(self):
        transformer = FilterSources(
            self.stream, sources=("targets", "features"))
        assert_equal(transformer.sources, ('features', 'targets'))

    def test_raises_value_error_on_nonexistent_sources(self):
        assert_raises(
            ValueError, FilterSources, self.stream, ['error', 'targets'])

    def test_filters_axis_labels(self):
        transformer = FilterSources(self.stream, sources=("features",))
        assert_equal(transformer.axis_labels,
                     {'features': ('batch', 'width', 'height')})


class VerifyWarningHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.records = []
        super(VerifyWarningHandler, self).__init__(*args, **kwargs)

    def handle(self, record):
        self.records.append(record)


class TestExpectsAxisLabels(object):
    def setUp(self):
        self.obj = ExpectsAxisLabels()
        self.handler = VerifyWarningHandler()
        logging.getLogger().addHandler(self.handler)

    def tearDown(self):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, VerifyWarningHandler):
                root_logger.handlers.remove(handler)

    def test_warning(self):
        self.obj.verify_axis_labels(('a', 'b', 'c'), None, 'foo')
        assert len(self.handler.records) == 1
        assert self.handler.records[0].levelno == logging.WARNING

    def test_exception(self):
        assert_raises(ValueError, self.obj.verify_axis_labels, ('a', 'b', 'c'),
                      ('b', 'c', 'd'), 'foo')


class TestToBytes(object):
    def setUp(self):
        self.string_data = [b'Hello', b'World!']
        self.dataset = IndexableDataset(
            indexables={'words': [numpy.fromstring(s, dtype='uint8')
                                  for s in self.string_data]},
            axis_labels={'words': ('batch', 'bytes')})

    def test_examplewise(self):
        stream = DataStream(
            dataset=self.dataset, iteration_scheme=SequentialExampleScheme(2))
        decoded_stream = ToBytes(stream)
        assert_equal(self.string_data,
                     [s for s, in decoded_stream.get_epoch_iterator()])

    def test_batchwise(self):
        stream = DataStream(
            dataset=self.dataset, iteration_scheme=SequentialScheme(2, 2))
        decoded_stream = ToBytes(stream)
        assert_equal([self.string_data],
                     [s for s, in decoded_stream.get_epoch_iterator()])
