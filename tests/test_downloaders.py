import hashlib
import os
import shutil
import tempfile
from unittest import TestCase

from fuel.downloaders.base import (download, default_downloader,
                                   filename_from_url)

iris_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
            'iris/iris.data')
iris_hash = "42615765a885ddf54427f12c34a0a070"


class DummyArgs:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def test_filename_from_url():
    assert filename_from_url(iris_url) == 'iris.data'


def test_download():
    f = tempfile.SpooledTemporaryFile()
    download(iris_url, f)
    f.seek(0)
    assert hashlib.md5(f.read()).hexdigest() == iris_hash
    f.close()


class TestDefaultDownloader(TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_default_downloader_save_with_filename(self):
        iris_path = os.path.join(self.tempdir, 'iris.data')
        args = DummyArgs(directory=self.tempdir, clear=False, urls=[iris_url],
                         filenames=['iris.data'])
        default_downloader(args)
        with open(iris_path, 'r') as f:
            assert hashlib.md5(
                f.read().encode('utf-8')).hexdigest() == iris_hash
        os.remove(iris_path)

    def test_default_downloader_save_no_filename(self):
        iris_path = os.path.join(self.tempdir, 'iris.data')
        args = DummyArgs(directory=self.tempdir, clear=False, urls=[iris_url],
                         filenames=[None])
        default_downloader(args)
        with open(iris_path, 'r') as f:
            assert hashlib.md5(
                f.read().encode('utf-8')).hexdigest() == iris_hash
        os.remove(iris_path)

    def test_default_downloader_clear(self):
        file_path = os.path.join(self.tempdir, 'tmp.data')
        open(file_path, 'a').close()
        args = DummyArgs(directory=self.tempdir, clear=True, urls=[None],
                         filenames=['tmp.data'])
        default_downloader(args)
        assert not os.path.isfile(file_path)
