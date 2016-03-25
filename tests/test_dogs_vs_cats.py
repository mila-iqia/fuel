import os
import shutil
import tempfile
import zipfile

import h5py
import numpy
import six
from PIL import Image
from numpy.testing import assert_raises

from fuel import config
from fuel.converters.dogs_vs_cats import convert_dogs_vs_cats
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


def setup():
    config._old_data_path = config.data_path
    config.data_path = tempfile.mkdtemp()
    _make_dummy_data(config.data_path[0])


def _make_dummy_data(output_directory):
    data = six.BytesIO()
    Image.new('RGB', (1, 1)).save(data, 'JPEG')
    image = data.getvalue()

    output_files = [os.path.join(output_directory,
                                 'dogs_vs_cats.{}.zip'.format(set_))
                    for set_ in ['train', 'test1']]
    with zipfile.ZipFile(output_files[0], 'w') as zip_file:
        zif = zipfile.ZipInfo('train/')
        zip_file.writestr(zif, "")
        for i in range(25000):
            zip_file.writestr('train/cat.{}.jpeg'.format(i), image)
    with zipfile.ZipFile(output_files[1], 'w') as zip_file:
        zif = zipfile.ZipInfo('test1/')
        zip_file.writestr(zif, "")
        for i in range(12500):
            zip_file.writestr('test1/{}.jpeg'.format(i), image)


def teardown():
    shutil.rmtree(config.data_path[0])
    config.data_path = config._old_data_path
    del config._old_data_path


def test_dogs_vs_cats():
    _test_conversion()
    _test_dataset()


def _test_conversion():
    convert_dogs_vs_cats(config.data_path[0], config.data_path[0])
    output_file = "dogs_vs_cats.hdf5"
    output_file = os.path.join(config.data_path[0], output_file)
    with h5py.File(output_file, 'r') as h5:
        assert numpy.all(h5['targets'][:25000] == 0)
        assert numpy.all(h5['targets'][25000:] == 1)
        assert numpy.all(numpy.array(
            [img for img in h5['image_features'][:]]) == 0)
        assert numpy.all(h5['image_features_shapes'][:, 0] == 3)
        assert numpy.all(h5['image_features_shapes'][:, 1:] == 1)


def _test_dataset():
    train = DogsVsCats(('train',))
    assert train.num_examples == 25000
    assert_raises(ValueError, DogsVsCats, ('valid',))

    test = DogsVsCats(('test',))
    stream = DataStream.default_stream(
        test, iteration_scheme=SequentialScheme(10, 10))
    data = next(stream.get_epoch_iterator())[0][0]
    assert data.dtype.kind == 'f'


test_dogs_vs_cats.setup = setup
test_dogs_vs_cats.teardown = teardown
