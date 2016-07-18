import numpy

from fuel.datasets import MJSynth
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from tests import skip_if_not_available


def test_mjsynth():
    skip_if_not_available(datasets=['mjsynth.hdf5'])

    train = MJSynth(('train',), load_in_memory=False)
    assert train.num_examples == 7224586
    handle = train.open()
    features, targets = train.get_data(handle, slice(7224576, 7224586))

    assert features.shape[:1] == (10,)
    assert targets.shape[:1] == (10,)
    train.close(handle)

    test = MJSynth(('test',), load_in_memory=False)
    assert test.num_examples == 891924
    handle = test.open()
    features, targets = test.get_data(handle, slice(0, 15))

    assert features.shape[:1] == (15,)
    assert targets.shape[:1] == (15,)

    assert features[0].dtype == numpy.uint8
    assert targets[0].dtype == numpy.dtype('S1')

    test.close(handle)

    val = MJSynth(('val',), load_in_memory=False)
    assert val.num_examples == 802731
    handle = val.open()
    features, targets = val.get_data(handle, slice(49990, 50000))

    assert features.shape[:1] == (10,)
    assert targets.shape[:1] == (10,)
    val.close(handle)

    stream = DataStream.default_stream(
        test, iteration_scheme=SequentialScheme(10, 10))
    features = next(stream.get_epoch_iterator())[0]

    def test_feature(feature):
        assert feature.min() >= 0.0 and feature.max() <= 1.0

    [test_feature(feature) for feature in features]
