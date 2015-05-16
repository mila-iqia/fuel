import argparse
import os
import shutil
import tempfile

from httmock import all_requests, response, HTTMock

from numpy.testing import assert_equal, assert_raises

from fuel.downloaders import mnist, binarized_mnist, cifar10
from fuel.downloaders.base import (download, default_downloader,
                                   filename_from_url, NeedURLPrefix)

mock_url = 'http://mock.com/mock.data'
mock_filename = 'mock.data'
mock_content = b'mock'


@all_requests
def response_content(url, request):
    """Mocks an HTTP response."""
    return {'status_code': 200, 'content': b'mock'}

@all_requests
def response_content_disposition(url, request):
    """Mocks an HTTP response with a content-disposition header."""
    headers = {
        'content-disposition': 'attachment; filename={}'.format(mock_filename)}
    return response(status_code=200, content=mock_content, headers=headers)


class TestFilenameFromURL(object):
    def test_no_content_disposition(self):
        with HTTMock(response_content):
            assert_equal(filename_from_url(mock_url), mock_filename)

    def test_content_disposition(self):
        with HTTMock(response_content_disposition):
            assert_equal(filename_from_url(mock_url), mock_filename)


class TestDownload(object):
    def test_download_content(self):
        with HTTMock(response_content), tempfile.SpooledTemporaryFile() as f:
            download(mock_url, f)
            f.seek(0)
            assert_equal(f.read(), mock_content)


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
        self.filepath = os.path.join(self.tempdir, mock_filename)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_default_downloader_save_with_filename(self):
        args = dict(directory=self.tempdir, clear=False, urls=[mock_url],
                    filenames=[mock_filename])
        with HTTMock(response_content):
            default_downloader(**args)
        with open(self.filepath, 'rb') as f:
            assert_equal(f.read(), mock_content)

    def test_default_downloader_save_no_filename(self):
        args = dict(directory=self.tempdir, clear=False, urls=[mock_url],
                    filenames=[None])
        with HTTMock(response_content):
            default_downloader(**args)
        with open(self.filepath, 'rb') as f:
            assert_equal(f.read(), mock_content)

    def test_default_downloader_save_no_url_url_prefix(self):
        args = dict(directory=self.tempdir, clear=False, urls=[None],
                    filenames=[mock_filename], url_prefix=mock_url[:-9])
        with HTTMock(response_content):
            default_downloader(**args)
        with open(self.filepath, 'rb') as f:
            assert_equal(f.read(), mock_content)

    def test_default_downloader_save_no_url_no_url_prefix(self):
        args = dict(directory=self.tempdir, clear=False, urls=[None],
                    filenames=[mock_filename])
        assert_raises(NeedURLPrefix, default_downloader, **args)

    def test_default_downloader_save_no_filename_for_url(self):
        args = dict(directory=self.tempdir, clear=False, urls=[mock_url[:-9]],
                    filenames=[None])
        with HTTMock(response_content):
            assert_raises(ValueError, default_downloader, **args)

    def test_default_downloader_clear(self):
        file_path = os.path.join(self.tempdir, 'tmp.data')
        open(file_path, 'a').close()
        args = dict(directory=self.tempdir, clear=True, urls=[None],
                    filenames=['tmp.data'])
        with HTTMock(response_content):
            default_downloader(**args)
        assert not os.path.isfile(file_path)
