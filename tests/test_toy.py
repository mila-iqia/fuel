
from fuel.datasets.toy import Spiral


def test_spiral():
    ds = Spiral(num_examples=1000, classes=2)

    features, position, label = ds.get_data(None, slice(0, 1000))

    assert features.shape[0] == 1000
    assert position.shape[0] == 1000
    assert label.shape[0] == 1000

    assert features.max() <= 1.
    assert position.max() <= 1.
    assert label.max() == 2
