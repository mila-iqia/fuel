import os
import pickle
import tables

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises
from six.moves import range

from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset
from fuel.streams import DataStream


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


class TestH5PYDataset(object):
    def setUp(self):
        self.features = numpy.arange(3600, dtype='uint8').reshape((100, 36))
        self.targets = numpy.arange(30, dtype='uint8').reshape((30, 1))
        self.h5file = h5py.File(
            'file.hdf5', mode='w', driver='core', backing_store=False)
        self.h5file['features'] = self.features
        self.h5file['features'].dims[0].label = 'batch'
        self.h5file['features'].dims[1].label = 'feature'
        self.h5file['targets'] = self.targets
        self.h5file['targets'].dims[0].label = 'batch'
        self.h5file['targets'].dims[1].label = 'index'
        split_dict = {'train': {'features': (0, 20, '.'), 'targets': (0, 20)},
                      'test': {'features': (20, 30, ''), 'targets': (20, 30)},
                      'unlabeled': {'features': (30, 100)}}
        self.h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    def tearDown(self):
        self.h5file.close()

    def test_split_parsing(self):
        train_set = H5PYDataset(self.h5file, which_set='train')
        assert train_set.provides_sources == ('features', 'targets')
        test_set = H5PYDataset(self.h5file, which_set='test')
        assert test_set.provides_sources == ('features', 'targets')
        unlabeled_set = H5PYDataset(self.h5file, which_set='unlabeled')
        assert unlabeled_set.provides_sources == ('features',)

    def test_axis_labels(self):
        dataset = H5PYDataset(self.h5file, which_set='train')
        assert dataset.axis_labels == {'features': ('batch', 'feature'),
                                       'targets': ('batch', 'index')}

    def test_pickling(self):
        pickle.loads(pickle.dumps(H5PYDataset(self.h5file, which_set='train')))

    def test_data_stream_pickling(self):
        stream = DataStream(H5PYDataset(self.h5file, which_set='train'))
        pickle.loads(pickle.dumps(stream))
        stream.close()

    def test_multiple_instances(self):
        dataset_1 = H5PYDataset(self.h5file, which_set='train')
        dataset_2 = H5PYDataset(self.h5file, which_set='train')
        handle_1 = dataset_1.open()
        handle_2 = dataset_2.open()
        dataset_1.get_data(state=handle_1, request=slice(0, 10))
        dataset_2.get_data(state=handle_2, request=slice(0, 10))
        dataset_1.close(handle_1)
        dataset_2.close(handle_2)

    def test_split(self):
        train_set = H5PYDataset(self.h5file, which_set='train')
        test_set = H5PYDataset(self.h5file, which_set='test')
        train_handle = train_set.open()
        test_handle = test_set.open()
        assert_equal(train_set.get_data(train_handle, slice(0, 8)),
                     (self.features[:8], self.targets[:8]))
        assert_equal(test_set.get_data(test_handle, slice(0, 2)),
                     (self.features[20:22], self.targets[20:22]))
        train_set.close(train_handle)
        test_set.close(test_handle)

    def test_out_of_memory(self):
        dataset = H5PYDataset(
            self.h5file, which_set='test', load_in_memory=False)
        handle = dataset.open()
        assert_equal(dataset.get_data(handle, slice(3, 5)),
                     (self.features[23:25], self.targets[23:25]))
        dataset.close(handle)

    def test_in_memory(self):
        dataset = H5PYDataset(
            self.h5file, which_set='train', load_in_memory=True)
        handle = dataset.open()
        request = slice(0, 10)
        assert_equal(dataset.get_data(handle, request),
                     (self.features[request], self.targets[request]))
        dataset.close(handle)

    def test_out_of_memory_sorted_indices(self):
        dataset = H5PYDataset(
            self.h5file, which_set='train', load_in_memory=False,
            sort_indices=True)
        handle = dataset.open()
        request = [7, 4, 6, 2, 5]
        assert_equal(dataset.get_data(handle, request),
                     (self.features[request], self.targets[request]))
        dataset.close(handle)

    def test_out_of_memory_unsorted_indices(self):
        dataset = H5PYDataset(
            self.h5file, which_set='train', load_in_memory=False,
            sort_indices=False)
        handle = dataset.open()
        assert_raises(TypeError, dataset.get_data, handle, [7, 4, 6, 2, 5])
        dataset.close(handle)
