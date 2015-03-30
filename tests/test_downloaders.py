import hashlib
import os

from fuel.downloaders.base import download, default_manager

iris_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
            'iris/iris.data')
iris_hash = "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"


def test_download_no_path():
    download(iris_url)
    with open('iris.data', 'r') as f:
        assert hashlib.sha256(f.read()).hexdigest() == iris_hash
    os.remove('iris.data')


def test_download_path_is_dir():
    os.mkdir('tmp')
    download(iris_url, 'tmp')
    with open('tmp/iris.data', 'r') as f:
        assert hashlib.sha256(f.read()).hexdigest() == iris_hash
    os.remove('tmp/iris.data')
    os.rmdir('tmp')


def test_download_path_is_file():
    download(iris_url, 'iris_tmp.data')
    with open('iris_tmp.data', 'r') as f:
        assert hashlib.sha256(f.read()).hexdigest() == iris_hash
    os.remove('iris_tmp.data')


def test_default_manager_save():
    class DummyArgs:
        pass
    args = DummyArgs()
    args.directory = '.'
    args.clear = False
    default_manager([iris_url], ['iris.data'])(args)
    with open('iris.data', 'r') as f:
        assert hashlib.sha256(f.read()).hexdigest() == iris_hash
    os.remove('iris.data')


def test_default_manager_clear():
    open('tmp.data', 'a').close()
    class DummyArgs:
        pass
    args = DummyArgs()
    args.directory = '.'
    args.clear = True
    default_manager([None], ['tmp.data'])(args)
    assert not os.path.isfile('tmp.data')
