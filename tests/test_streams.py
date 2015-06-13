import unittest

import numpy
from numpy.testing import assert_equal

from fuel.datasets import IterableDataset
from fuel.streams import DataStream


class TestDataStream(unittest.TestCase):
    def setUp(self):
        self.dataset = IterableDataset(dict(data=numpy.eye(2),
                                            targets=numpy.arange(2)))

    def test_sources_setter(self):
        stream = DataStream(self.dataset)
        stream.sources = ('features',)
        assert_equal(stream.sources, ('features',))

    def test_sources_selection(self):
        stream = DataStream(self.dataset, sources=('data',))
        assert len(stream.get_epoch_iterator().next()) == 1
        stream = DataStream(self.dataset, sources=('data', 'targets'))
        assert len(stream.get_epoch_iterator().next()) == 2
        stream = DataStream(self.dataset, sources=('data', 'targets', 'error'))
        self.assertRaises(ValueError,
                          lambda: stream.get_epoch_iterator().next())

    def test_no_axis_labels(self):
        stream = DataStream(self.dataset)
        assert stream.axis_labels is None

    def test_axis_labels(self):
        axis_labels = {'data': ('batch', 'features')}
        self.dataset.axis_labels = axis_labels
        stream = DataStream(self.dataset)
        assert_equal(stream.axis_labels, axis_labels)
