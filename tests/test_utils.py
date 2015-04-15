import pickle
from six.moves import range

from numpy.testing import assert_raises

from fuel.utils import do_not_pickle_attributes, expand_axis_label


@do_not_pickle_attributes("non_pickable", "bulky_attr")
class TestClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.bulky_attr = list(range(100))
        self.non_pickable = lambda x: x


def test_do_not_pickle_attributes():
    cl = TestClass()

    dump = pickle.dumps(cl)

    loaded = pickle.loads(dump)
    assert loaded.bulky_attr == list(range(100))
    assert loaded.non_pickable is not None


def test_expand_axis_label():
    assert expand_axis_label('b') == 'batch'
    assert expand_axis_label('c') == 'channel'
    assert expand_axis_label('t') == 'time'
    assert expand_axis_label('0') == 'axis_0'
    assert expand_axis_label('1') == 'axis_1'
    assert expand_axis_label('0b') == '0b'
    assert expand_axis_label('') == ''


def test_expand_axis_label_not_string():
    assert_raises(ValueError, expand_axis_label, 0)
