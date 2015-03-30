import os

from fuel.downloaders.base import download, default_manager

iris_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
            'iris/iris.data')
iris_first_line = '5.1,3.5,1.4,0.2,Iris-setosa\n'


def test_download_no_path():
    download(iris_url)
    with open('iris.data') as f:
        first_line = f.readline()
    assert first_line == iris_first_line
    os.remove('iris.data')


def test_download_path_is_dir():
    os.mkdir('tmp')
    download(iris_url, 'tmp')
    with open('tmp/iris.data') as f:
        first_line = f.readline()
    assert first_line == iris_first_line
    os.remove('tmp/iris.data')
    os.rmdir('tmp')


def test_download_path_is_file():
    download(iris_url, 'iris_tmp.data')
    with open('iris_tmp.data') as f:
        first_line = f.readline()
    assert first_line == iris_first_line
    os.remove('iris_tmp.data')


def test_default_manager_save():
    class DummyArgs:
        pass
    args = DummyArgs()
    args.directory = '.'
    args.clear = False
    default_manager([iris_url], ['iris.data'])(args)
    with open('iris.data') as f:
        first_line = f.readline()
    assert first_line == iris_first_line
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
