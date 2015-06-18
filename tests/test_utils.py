import os
import shutil
import tempfile

from numpy.testing import assert_raises, assert_equal
from six.moves import range, cPickle

from fuel import config
from fuel.iterator import DataIterator
from fuel.utils import do_not_pickle_attributes, find_in_data_path


@do_not_pickle_attributes("non_picklable", "bulky_attr")
class DummyClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.bulky_attr = list(range(100))
        self.non_picklable = lambda x: x


class FaultyClass(object):
    pass


@do_not_pickle_attributes("iterator")
class UnpicklableClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.iterator = DataIterator(None)


@do_not_pickle_attributes("attribute")
class NonLoadingClass(object):
    def load(self):
        pass


class TestFindInDataPath(object):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tempdir, 'dir1'))
        os.mkdir(os.path.join(self.tempdir, 'dir2'))
        self.original_data_path = config.data_path
        config.data_path = os.path.pathsep.join(
            [os.path.join(self.tempdir, 'dir1'),
             os.path.join(self.tempdir, 'dir2')])
        with open(os.path.join(self.tempdir, 'dir1', 'file_1.txt'), 'w'):
            pass
        with open(os.path.join(self.tempdir, 'dir2', 'file_1.txt'), 'w'):
            pass
        with open(os.path.join(self.tempdir, 'dir2', 'file_2.txt'), 'w'):
            pass

    def tearDown(self):
        config.data_path = self.original_data_path
        shutil.rmtree(self.tempdir)

    def test_returns_file_path(self):
        assert_equal(find_in_data_path('file_2.txt'),
                     os.path.join(self.tempdir, 'dir2', 'file_2.txt'))

    def test_returns_first_file_found(self):
        assert_equal(find_in_data_path('file_1.txt'),
                     os.path.join(self.tempdir, 'dir1', 'file_1.txt'))

    def test_raises_error_on_file_not_found(self):
        assert_raises(IOError, find_in_data_path, 'dummy.txt')


class TestDoNotPickleAttributes(object):
    def test_load(self):
        instance = cPickle.loads(cPickle.dumps(DummyClass()))
        assert_equal(instance.bulky_attr, list(range(100)))
        assert instance.non_picklable is not None

    def test_value_error_no_load_method(self):
        assert_raises(ValueError, do_not_pickle_attributes("x"), FaultyClass)

    def test_value_error_iterator(self):
        assert_raises(ValueError, cPickle.dumps, UnpicklableClass())

    def test_value_error_attribute_non_loaded(self):
        assert_raises(ValueError, getattr, NonLoadingClass(), 'attribute')
