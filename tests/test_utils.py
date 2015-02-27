import pickle
from six.moves import range

from fuel.utils import do_not_pickle_attributes


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
