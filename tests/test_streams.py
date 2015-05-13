import numpy
from numpy.testing import assert_equal

from fuel.datasets import IterableDataset
from fuel.streams import DataStream


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

    def test_axis_labels(self):
        axis_labels = {'data': ('batch', 'features')}
        self.dataset.axis_labels = axis_labels
        stream = DataStream(self.dataset)
        assert_equal(stream.axis_labels, axis_labels)
