import numpy

from numpy.testing import assert_raises, assert_equal, assert_allclose

from fuel.datasets import Adult
from tests import skip_if_not_available


def test_adult_test():
    skip_if_not_available(datasets=['adult.hdf5'])

    dataset = Adult(('test',), load_in_memory=False)
    handle = dataset.open()
    data, labels = dataset.get_data(handle, slice(0, 10))

    assert data.shape == (10, 104)
    assert labels.shape == (10, 1)
    known = numpy.array(
        [25.,  38.,  28.,  44.,  34.,  63.,  24.,  55.,  65.,  36.])
    assert_allclose(data[:, 0], known)
    assert dataset.num_examples == 15060
    dataset.close(handle)

    dataset = Adult(('train',), load_in_memory=False)
    handle = dataset.open()
    data, labels = dataset.get_data(handle, slice(0, 10))

    assert data.shape == (10, 104)
    assert labels.shape == (10, 1)
    known = numpy.array(
        [39.,  50.,  38.,  53.,  28.,  37.,  49.,  52.,  31.,  42.])
    assert_allclose(data[:, 0], known)
    assert dataset.num_examples == 30162
    dataset.close(handle)


def test_adult_axes():
    skip_if_not_available(datasets=['adult.hdf5'])

    dataset = Adult(('test',), load_in_memory=False)
    assert_equal(dataset.axis_labels['features'],
                 ('batch', 'feature'))

    dataset = Adult(('train',), load_in_memory=False)
    assert_equal(dataset.axis_labels['features'],
                 ('batch', 'feature'))


def test_adult_invalid_split():
    skip_if_not_available(datasets=['adult.hdf5'])

    assert_raises(ValueError, Adult, ('dummy',))
