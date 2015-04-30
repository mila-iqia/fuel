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


def test_h5py_dataset_split_parsing():
    try:
        h5file = h5py.File('tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (100, 36), dtype='uint8')
        features[...] = numpy.zeros(shape=(100, 36)).astype('uint8')
        targets = h5file.create_dataset('targets', (30, 1), dtype='uint8')
        targets[...] = numpy.zeros(shape=(30, 1)).astype('uint8')
        split_dict = {'train': {'features': (0, 20), 'targets': (0, 20)},
                      'test': {'features': (20, 30), 'targets': (20, 30)},
                      'unlabeled': {'features': (30, 100)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        train_set = H5PYDataset(path='tmp.hdf5', which_set='train')
        assert train_set.provides_sources == ('features', 'targets')
        test_set = H5PYDataset(path='tmp.hdf5', which_set='test')
        assert test_set.provides_sources == ('features', 'targets')
        unlabeled_set = H5PYDataset(path='tmp.hdf5', which_set='unlabeled')
        assert unlabeled_set.provides_sources == ('features',)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_axis_labels():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features.dims[0].label = 'batch'
        features.dims[1].label = 'feature'
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(path='tmp.hdf5', which_set='train')
        assert dataset.axis_labels == {'features': ('batch', 'feature')}
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_pickles():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(path='tmp.hdf5', which_set='train')
        pickle.loads(pickle.dumps(dataset))
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_multiple_instances():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset_1 = H5PYDataset(path='tmp.hdf5', which_set='train')
        dataset_2 = H5PYDataset(path='tmp.hdf5', which_set='train')
        handle_1 = dataset_1.open()
        handle_2 = dataset_2.open()
        dataset_1.get_data(state=handle_1, request=slice(0, 10))
        dataset_2.get_data(state=handle_2, request=slice(0, 10))
        dataset_1.close(handle_1)
        dataset_2.close(handle_2)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_split():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 8)},
                      'test': {'features': (8, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
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
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_out_of_memory():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        targets = h5file.create_dataset('targets', (10, 1), dtype='float32')
        targets[...] = numpy.arange(10, dtype='float32').reshape((10, 1))
        split_dict = {'train': {'features': (0, 5), 'targets': (0, 5)},
                      'test': {'features': (5, 10), 'targets': (5, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(
            path='tmp.hdf5', which_set='test', load_in_memory=False)
        handle = dataset.open()
        assert_equal(
            dataset.get_data(state=handle, request=slice(3, 5))[1],
            numpy.arange(10).reshape((10, 1))[8:10])
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_in_memory():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(
            path='tmp.hdf5', which_set='train', load_in_memory=True)
        handle = dataset.open()
        assert_equal(
            dataset.get_data(state=handle, request=slice(0, 10))[0],
            numpy.arange(50).reshape((10, 5)))
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_flatten_in_memory():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset(
            'features', (10, 2, 3), dtype='float32')
        features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
        targets = h5file.create_dataset('targets', (10,), dtype='uint8')
        targets[...] = numpy.arange(10, dtype='uint8')
        split_dict = {'train': {'features': (0, 10), 'targets': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(which_set='train', path='tmp.hdf5',
                              load_in_memory=True, flatten=['features'])
        handle = dataset.open()
        assert_equal(
            dataset.get_data(state=handle, request=slice(0, 10))[0],
            numpy.arange(60).reshape((10, 6)))
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_flatten_out_of_memory():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset(
            'features', (10, 2, 3), dtype='float32')
        features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
        targets = h5file.create_dataset('targets', (10,), dtype='uint8')
        targets[...] = numpy.arange(10, dtype='uint8')
        split_dict = {'train': {'features': (0, 10), 'targets': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(path='tmp.hdf5', load_in_memory=False,
                              which_set='train', flatten=['features'])
        handle = dataset.open()
        assert_equal(
            dataset.get_data(state=handle, request=slice(0, 10))[0],
            numpy.arange(60).reshape((10, 6)))
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_out_of_memory_sorted_indices():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(
            path='tmp.hdf5', which_set='train', load_in_memory=False,
            sort_indices=True)
        handle = dataset.open()
        assert_equal(
            dataset.get_data(state=handle, request=[7, 4, 6, 2, 5])[0],
            numpy.arange(50).reshape((10, 5))[[7, 4, 6, 2, 5]])
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_dataset_out_of_memory_unsorted_indices():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset('features', (10, 5), dtype='float32')
        features[...] = numpy.arange(50, dtype='float32').reshape((10, 5))
        split_dict = {'train': {'features': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(
            path='tmp.hdf5', which_set='train', load_in_memory=False,
            sort_indices=False)
        handle = dataset.open()
        assert_raises(TypeError, dataset.get_data, handle, [7, 4, 6, 2, 5])
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')


def test_h5py_flatten_raises_error_on_invalid_name():
    try:
        h5file = h5py.File(name='tmp.hdf5', mode="w")
        features = h5file.create_dataset(
            'features', (10, 2, 3), dtype='float32')
        features[...] = numpy.arange(60, dtype='float32').reshape((10, 2, 3))
        targets = h5file.create_dataset('targets', (10,), dtype='uint8')
        targets[...] = numpy.arange(10, dtype='uint8')
        split_dict = {'train': {'features': (0, 10), 'targets': (0, 10)}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5file.flush()
        h5file.close()
        dataset = H5PYDataset(path='tmp.hdf5', load_in_memory=False,
                              which_set='train', flatten=['features'])
        handle = dataset.open()
        assert_raises(
            ValueError, H5PYDataset, 'tmp.hdf5',
            None, None, False, 'foo', None)
        dataset.close(handle)
    finally:
        if os.path.exists('tmp.hdf5'):
            os.remove('tmp.hdf5')
