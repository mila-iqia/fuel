import numpy
from numpy.testing import assert_raises

from fuel import config
from fuel.datasets import CalTech101Silhouettes
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


def test_caltech101_silhouettes():
    for which_set, size, num_examples in (
            ('train', 16, 4082), ('valid', 16, 2257), ('test', 16, 2302),
            ('train', 16, 4082), ('valid', 16, 2257), ('test', 16, 2302)):
        ds = CalTech101Silhouettes(which_set=which_set, size=size, load_in_memory=False)

        assert ds.num_examples == num_examples

        handle = ds.open()
        features, targets = ds.get_data(handle, slice(0, 10))
    
        assert features.shape == (10, 1, size, size)
        assert targets.shape == (10, 1)

        assert features.dtype == numpy.uint8
        assert targets.dtype == numpy.uint8

    assert_raises(ValueError, CalTech101Silhouettes, which_set='test', size=10)
