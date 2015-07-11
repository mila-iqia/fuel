import numpy
from numpy.testing import assert_equal, assert_raises

from fuel.datasets import IterableDataset, IndexableDataset
from fuel.schemes import SequentialExampleScheme, SequentialScheme
from fuel.streams import AbstractDataStream, DataStream


class DummyDataStream(AbstractDataStream):
    def reset(self):
        pass

    def close(self):
        pass

    def next_epoch(self):
        pass

    def get_epoch_iterator(self, as_dict=False):
        pass

    def get_data(self, request=None):
        pass


class TestAbstractDataStream(object):
    def test_raises_value_error_on_no_scheme_no_produces_examples(self):
        stream = DummyDataStream()
        assert_raises(ValueError, getattr, stream, 'produces_examples')

    def test_raises_value_error_when_setting_produces_examples_if_scheme(self):
        stream = DummyDataStream(SequentialExampleScheme(2))
        assert_raises(ValueError, setattr, stream, 'produces_examples', True)


class TestDataStream(object):
    def setUp(self):
        self.dataset = IterableDataset(numpy.eye(2))

    def test_sources_setter(self):
        stream = DataStream(self.dataset)
        stream.sources = ('features',)
        assert_equal(stream.sources, ('features',))

    def test_no_axis_labels(self):
        stream = DataStream(self.dataset)
        assert stream.axis_labels is None

    def test_axis_labels_on_produces_examples(self):
        axis_labels = {'data': ('batch', 'features')}
        self.dataset.axis_labels = axis_labels
        stream = DataStream(self.dataset)
        assert_equal(stream.axis_labels, {'data': ('features',)})

    def test_axis_labels_on_produces_batches(self):
        dataset = IndexableDataset(numpy.eye(2))
        axis_labels = {'data': ('batch', 'features')}
        dataset.axis_labels = axis_labels
        stream = DataStream(dataset, iteration_scheme=SequentialScheme(2, 2))
        assert_equal(stream.axis_labels, axis_labels)

    def test_produces_examples(self):
        stream = DataStream(self.dataset,
                            iteration_scheme=SequentialExampleScheme(2))
        assert stream.produces_examples
