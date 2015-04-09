import h5py
import numpy
from numpy.testing import assert_equal, assert_raises

from fuel.converters.base import fill_hdf5_file


def test_fill_hdf5_file():
    h5file = h5py.File(
        'tmp.hdf5', mode="w", driver='core', backing_store=False)
    train_features = numpy.arange(16, dtype='uint8').reshape((4, 2, 2))
    test_features = numpy.arange(8, dtype='uint8').reshape((2, 2, 2)) + 3
    train_targets = numpy.arange(4, dtype='float32').reshape((4, 1))
    test_targets = numpy.arange(2, dtype='float32').reshape((2, 1)) + 3
    fill_hdf5_file(
        h5file,
        (('train', 'features', train_features),
         ('train', 'targets', train_targets),
         ('test', 'features', test_features),
         ('test', 'targets', test_targets)))
    assert_equal(
        h5file['features'], numpy.vstack([train_features, test_features]))
    assert_equal(
        h5file['targets'], numpy.vstack([train_targets, test_targets]))
    assert h5file['features'].dtype == 'uint8'
    assert h5file['targets'].dtype == 'float32'
    h5file.close()


def test_fill_hdf5_file_multi_length_error():
    h5file = h5py.File(
        'tmp.hdf5', mode="w", driver='core', backing_store=False)
    train_features = numpy.arange(16, dtype='uint8').reshape((4, 2, 2))
    train_targets = numpy.arange(8, dtype='float32').reshape((8, 1))
    assert_raises(
        ValueError, fill_hdf5_file, h5file,
        (('train', 'features', train_features),
         ('train', 'targets', train_targets)))
    h5file.close()


def test_fill_hdf5_file_multi_dtype_error():
    h5file = h5py.File(
        'tmp.hdf5', mode="w", driver='core', backing_store=False)
    train_features = numpy.arange(16, dtype='uint8').reshape((4, 2, 2))
    test_features = numpy.arange(8, dtype='float32').reshape((2, 2, 2)) + 3
    assert_raises(
        ValueError, fill_hdf5_file, h5file,
        (('train', 'features', train_features),
         ('test', 'features', test_features)))
    h5file.close()


def test_fill_hdf5_file_multi_shape_error():
    h5file = h5py.File(
        'tmp.hdf5', mode="w", driver='core', backing_store=False)
    train_features = numpy.arange(16, dtype='uint8').reshape((4, 2, 2))
    test_features = numpy.arange(16, dtype='float32').reshape((2, 4, 2)) + 3
    assert_raises(
        ValueError, fill_hdf5_file, h5file,
        (('train', 'features', train_features),
         ('test', 'features', test_features)))
    h5file.close()
