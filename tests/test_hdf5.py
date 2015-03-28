import os
import pickle
import tables

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises
from six.moves import range

from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset


def test_hdf5_dataset():
    num_rows = 500
    filters = tables.Filters(complib='blosc', complevel=5)

    h5file = tables.open_file("tmp.h5", mode="w", title="Test file",
                              filters=filters)
    group = h5file.create_group("/", 'Data')
    atom = tables.UInt8Atom()
    y = h5file.create_carray(group, 'y', atom=atom, title='Data targets',
                             shape=(num_rows, 1), filters=filters)
    for i in range(num_rows):
        y[i] = i
    h5file.flush()
    h5file.close()

    dataset = Hdf5Dataset(['y'], 0, 500, 'tmp.h5')
    assert_equal(dataset.get_data(request=slice(0, 10))[0],
                 numpy.arange(10).reshape(10, 1))
    # Test if pickles
    dump = pickle.dumps(dataset)
    pickle.loads(dump)

    os.remove('tmp.h5')


def test_h5py_dataset_pickles():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset('features', (10, 5), dtype='float32')
    features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(path='tmp.hdf5')
    pickle.loads(pickle.dumps(dataset))
    os.remove('tmp.hdf5')


def test_h5py_dataset_multiple_instances():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset('features', (10, 5), dtype='float32')
    features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
    h5file.flush()
    h5file.close()
    dataset_1 = H5PYDataset(path='tmp.hdf5')
    dataset_2 = H5PYDataset(path='tmp.hdf5')
    handle_1 = dataset_1.open()
    handle_2 = dataset_2.open()
    dataset_1.get_data(state=handle_1, request=slice(0, 10))
    dataset_2.get_data(state=handle_2, request=slice(0, 10))
    dataset_1.close(handle_1)
    dataset_2.close(handle_2)
    os.remove('tmp.hdf5')


def test_h5py_dataset_split():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset('features', (10, 5), dtype='float32')
    features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
    h5file.attrs['train'] = [0, 8]
    h5file.attrs['test'] = [8, 10]
    h5file.flush()
    h5file.close()
    train_set = H5PYDataset(path='tmp.hdf5', which_set='train')
    test_set = H5PYDataset(path='tmp.hdf5', which_set='test')
    train_handle = train_set.open()
    test_handle = test_set.open()
    assert_equal(
        train_set.get_data(state=train_handle, request=slice(0, 8))[0],
        numpy.arange(50).reshape((10, 5))[:8])
    assert_equal(
        test_set.get_data(state=test_handle, request=slice(0, 2))[0],
        numpy.arange(50).reshape((10, 5))[8:])
    train_set.close(train_handle)
    test_set.close(test_handle)
    os.remove('tmp.hdf5')


def test_h5py_dataset_out_of_memory():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset('features', (10, 5), dtype='float32')
    features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(path='tmp.hdf5', load_in_memory=False)
    handle = dataset.open()
    assert_equal(
        dataset.get_data(state=handle, request=slice(0, 10))[0],
        numpy.arange(50).reshape((10, 5)))
    dataset.close(handle)
    os.remove('tmp.hdf5')


def test_h5py_dataset_in_memory():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset('features', (10, 5), dtype='float32')
    features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(path='tmp.hdf5', load_in_memory=True)
    handle = dataset.open()
    assert_equal(
        dataset.get_data(state=handle, request=slice(0, 10))[0],
        numpy.arange(50).reshape((10, 5)))
    dataset.close(handle)
    os.remove('tmp.hdf5')


def test_h5py_flatten_in_memory():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset(
        'features', (10, 2, 3), dtype='float32')
    features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
    targets = h5file.create_dataset('targets', (10,), dtype='uint8')
    targets[...] = numpy.arange(10, dtype='uint8')
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(
        path='tmp.hdf5', load_in_memory=True, flatten=['features'])
    handle = dataset.open()
    assert_equal(
        dataset.get_data(state=handle, request=slice(0, 10))[0],
        numpy.arange(60).reshape((10, 6)))
    dataset.close(handle)
    os.remove('tmp.hdf5')


def test_h5py_flatten_out_of_memory():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset(
        'features', (10, 2, 3), dtype='float32')
    features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
    targets = h5file.create_dataset('targets', (10,), dtype='uint8')
    targets[...] = numpy.arange(10, dtype='uint8')
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(
        path='tmp.hdf5', load_in_memory=False, flatten=['features'])
    handle = dataset.open()
    assert_equal(
        dataset.get_data(state=handle, request=slice(0, 10))[0],
        numpy.arange(60).reshape((10, 6)))
    dataset.close(handle)
    os.remove('tmp.hdf5')


def test_h5py_flatten_raises_error_on_invalid_name():
    h5file = h5py.File(name='tmp.hdf5', mode="w")
    features = h5file.create_dataset(
        'features', (10, 2, 3), dtype='float32')
    features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
    targets = h5file.create_dataset('targets', (10,), dtype='uint8')
    targets[...] = numpy.arange(10, dtype='uint8')
    h5file.flush()
    h5file.close()
    dataset = H5PYDataset(
        path='tmp.hdf5', load_in_memory=False, flatten=['features'])
    handle = dataset.open()
    assert_raises(
        ValueError, H5PYDataset, 'tmp.hdf5', None, None, False, 'foo', None)
    dataset.close(handle)
    os.remove('tmp.hdf5')
