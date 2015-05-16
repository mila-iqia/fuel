import argparse
import hashlib
import os
import shutil
import tempfile

from numpy.testing import assert_equal, assert_raises

from fuel.downloaders import mnist, binarized_mnist, cifar10
from fuel.downloaders.base import (download, default_downloader,
                                   filename_from_url, NeedURLPrefix)

iris_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
            'iris/iris.data')
iris_hash = "42615765a885ddf54427f12c34a0a070"


def test_filename_from_url():
    assert filename_from_url(iris_url) == 'iris.data'


def test_download():
    f = tempfile.SpooledTemporaryFile()
    download(iris_url, f)
    f.seek(0)
    assert hashlib.md5(f.read()).hexdigest() == iris_hash
    f.close()


def test_mnist():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    mnist.fill_subparser(subparsers.add_parser('mnist'))
    args = parser.parse_args(['mnist'])
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    urls = ['http://yann.lecun.com/exdb/mnist/' + f for f in filenames]
    assert_equal(args.filenames, filenames)
    assert_equal(args.urls, urls)
    assert args.func is default_downloader


def test_binarized_mnist():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    binarized_mnist.fill_subparser(subparsers.add_parser('binarized_mnist'))
    args = parser.parse_args(['binarized_mnist'])
    sets = ['train', 'valid', 'test']
    urls = ['http://www.cs.toronto.edu/~larocheh/public/datasets/' +
            'binarized_mnist/binarized_mnist_{}.amat'.format(s) for s in sets]
    filenames = ['binarized_mnist_{}.amat'.format(s) for s in sets]
    assert_equal(args.filenames, filenames)
    assert_equal(args.urls, urls)
    assert args.func is default_downloader


def test_cifar10():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cifar10.fill_subparser(subparsers.add_parser('cifar10'))
    args = parser.parse_args(['cifar10'])
    filenames = ['cifar-10-python.tar.gz']
    urls = ['http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
    assert_equal(args.filenames, filenames)
    assert_equal(args.urls, urls)
    assert args.func is default_downloader


class TestDefaultDownloader(object):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_default_downloader_save_with_filename(self):
        iris_path = os.path.join(self.tempdir, 'iris.data')
        args = dict(directory=self.tempdir, clear=False, urls=[iris_url],
                    filenames=['iris.data'])
        default_downloader(**args)
        with open(iris_path, 'r') as f:
            assert hashlib.md5(
                f.read().encode('utf-8')).hexdigest() == iris_hash
        os.remove(iris_path)

    def test_default_downloader_save_no_filename(self):
        iris_path = os.path.join(self.tempdir, 'iris.data')
        args = dict(directory=self.tempdir, clear=False, urls=[iris_url],
                    filenames=[None])
        default_downloader(**args)
        with open(iris_path, 'r') as f:
            assert hashlib.md5(
                f.read().encode('utf-8')).hexdigest() == iris_hash
        os.remove(iris_path)

    def test_default_downloader_clear(self):
        file_path = os.path.join(self.tempdir, 'tmp.data')
        open(file_path, 'a').close()
        args = dict(directory=self.tempdir, clear=True, urls=[None],
                    filenames=['tmp.data'])
        default_downloader(**args)
        assert not os.path.isfile(file_path)
