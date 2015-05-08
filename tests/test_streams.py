import numpy

from fuel.datasets import IterableDataset
from fuel.streams import DataStream


def test_data_stream_no_axis_labels():
    dataset = IterableDataset(numpy.eye(2))
    stream = DataStream(dataset)
    assert stream.axis_labels is None


def test_data_stream_axis_labels():
    axis_labels = {'data': ('batch', 'features')}
    dataset = IterableDataset(numpy.eye(2), axis_labels=axis_labels)
    stream = DataStream(dataset)
    assert stream.axis_labels == axis_labels
