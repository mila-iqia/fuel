import numpy
from numpy.testing import assert_raises

from fuel.datasets import CalTech101Silhouettes
from tests import skip_if_not_available


def test_caltech101_silhouettes16():
    skip_if_not_available(datasets=['caltech101_silhouettes16.hdf5'])
    for which_set, size, num_examples in (
            ('train', 16, 4082), ('valid', 16, 2257), ('test', 16, 2302)):
        ds = CalTech101Silhouettes(which_sets=[which_set], size=size,
                                   load_in_memory=False)

        assert ds.num_examples == num_examples

        handle = ds.open()
        features, targets = ds.get_data(handle, slice(0, 10))

        assert features.shape == (10, 1, size, size)
        assert targets.shape == (10, 1)

        assert features.dtype == numpy.uint8
        assert targets.dtype == numpy.uint8


def test_caltech101_silhouettes_unkn_size():
    assert_raises(ValueError, CalTech101Silhouettes,
                  which_sets=['test'], size=10)


def test_caltech101_silhouettes28():
    skip_if_not_available(datasets=['caltech101_silhouettes28.hdf5'])
    for which_set, size, num_examples in (
            ('train', 28, 4100), ('valid', 28, 2264), ('test', 28, 2307)):
        ds = CalTech101Silhouettes(which_sets=[which_set], size=size,
                                   load_in_memory=False)

        assert ds.num_examples == num_examples

        handle = ds.open()
        features, targets = ds.get_data(handle, slice(0, 10))

        assert features.shape == (10, 1, size, size)
        assert targets.shape == (10, 1)

        assert features.dtype == numpy.uint8
        assert targets.dtype == numpy.uint8
