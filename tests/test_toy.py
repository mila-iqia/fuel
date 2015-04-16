
from fuel.datasets.toy import Spiral, SwissRoll


def test_spiral():
    ds = Spiral(num_examples=1000, classes=2)

    features, position, label = ds.get_data(None, slice(0, 1000))

    assert features.ndim == 2
    assert features.shape[0] == 1000
    assert features.shape[1] == 2

    assert position.ndim == 1
    assert position.shape[0] == 1000

    assert label.ndim == 1
    assert label.shape[0] == 1000

    assert features.max() <= 1.
    assert position.max() <= 1.
    assert label.max() == 1


def test_swiossroll():
    ds = SwissRoll(num_examples=1000)

    features, position = ds.get_data(None, slice(0, 1000))

    assert features.ndim == 2
    assert features.shape[0] == 1000
    assert features.shape[1] == 3

    assert position.ndim == 2
    assert position.shape[0] == 1000
    assert position.shape[1] == 2

    assert features.max() <= 1.
    assert position.max() <= 1.
