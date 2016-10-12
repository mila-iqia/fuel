import tarfile

import numpy
import six

from .test_convert_ilsvrc2010 import MockH5PYFile
# from fuel.server import recv_arrays, send_arrays
from fuel.converters.ilsvrc2012 import (prepare_hdf5_file,
                                        prepare_metadata,
                                        read_devkit,
                                        read_metadata_mat_file,
                                        DEVKIT_META_PATH,
                                        DEVKIT_ARCHIVE,
                                        TEST_IMAGES_TAR)
from fuel.utils import find_in_data_path
from tests import skip_if_not_available


def test_prepare_metadata():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE, TEST_IMAGES_TAR])
    devkit_path = find_in_data_path(DEVKIT_ARCHIVE)
    n_train, v_gt, n_test, wnid_map = prepare_metadata(devkit_path)
    assert n_train == 1281167
    assert len(v_gt) == 50000
    assert n_test == 100000
    assert sorted(wnid_map.values()) == list(range(1000))
    assert all(isinstance(k, six.string_types) and len(k) == 9
               for k in wnid_map)


def test_prepare_hdf5_file():
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, 10, 5, 2)

    def get_start_stop(hdf5_file, split):
        rows = [r for r in hdf5_file.attrs['split'] if
                (r['split'].decode('utf8') == split)]
        return dict([(r['source'].decode('utf8'), (r['start'], r['stop']))
                     for r in rows if r['stop'] - r['start'] > 0])

    # Verify properties of the train split.
    train_splits = get_start_stop(hdf5_file, 'train')
    assert all(v == (0, 10) for v in train_splits.values())
    assert set(train_splits.keys()) == set([u'encoded_images', u'targets',
                                            u'filenames'])

    # Verify properties of the valid split.
    valid_splits = get_start_stop(hdf5_file, 'valid')
    assert all(v == (10, 15) for v in valid_splits.values())
    assert set(valid_splits.keys()) == set([u'encoded_images', u'targets',
                                            u'filenames'])

    # Verify properties of the test split.
    test_splits = get_start_stop(hdf5_file, 'test')
    assert all(v == (15, 17) for v in test_splits.values())
    assert set(test_splits.keys()) == set([u'encoded_images', u'filenames'])

    from numpy import dtype

    # Verify properties of the encoded_images HDF5 dataset.
    assert hdf5_file['encoded_images'].shape[0] == 17
    assert len(hdf5_file['encoded_images'].shape) == 1
    assert hdf5_file['encoded_images'].dtype.kind == 'O'
    assert hdf5_file['encoded_images'].dtype.metadata['vlen'] == dtype('uint8')

    # Verify properties of the filenames dataset.
    assert hdf5_file['filenames'].shape[0] == 17
    assert len(hdf5_file['filenames'].shape) == 2
    assert hdf5_file['filenames'].dtype == dtype('S32')

    # Verify properties of the targets dataset.
    assert hdf5_file['targets'].shape[0] == 15
    assert hdf5_file['targets'].shape[1] == 1
    assert len(hdf5_file['targets'].shape) == 2
    assert hdf5_file['targets'].dtype == dtype('int16')


def test_read_devkit():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE])
    synsets, raw_valid_gt = read_devkit(find_in_data_path(DEVKIT_ARCHIVE))
    # synset sanity tests appear in test_read_metadata_mat_file
    assert raw_valid_gt.min() == 1
    assert raw_valid_gt.max() == 1000
    assert raw_valid_gt.dtype.kind == 'i'
    assert raw_valid_gt.shape == (50000,)


def test_read_metadata_mat_file():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE])
    with tarfile.open(find_in_data_path(DEVKIT_ARCHIVE)) as tar:
        meta_mat = tar.extractfile(DEVKIT_META_PATH)
        synsets = read_metadata_mat_file(meta_mat)
    assert (synsets['ILSVRC2012_ID'] ==
            numpy.arange(1, len(synsets) + 1)).all()
    assert synsets['num_train_images'][1000:].sum() == 0
    assert (synsets['num_train_images'][:1000] > 0).all()
    assert synsets.ndim == 1
    assert synsets['wordnet_height'].min() == 0
    assert synsets['wordnet_height'].max() == 19
    assert synsets['WNID'].dtype == numpy.dtype('S9')
    assert (synsets['num_children'][:1000] == 0).all()
    assert (synsets['children'][:1000] == -1).all()
