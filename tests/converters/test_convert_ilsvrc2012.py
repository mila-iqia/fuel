import io
import tarfile

import numpy
import six
import zmq
from numpy.testing import assert_equal

from test_convert_ilsvrc2010 import (create_fake_jpeg_tar,
                                     create_fake_tar_of_tars,
                                     MockH5PYFile,
                                     MockSocket,
                                     MOCK_CONSUMER_MESSAGES)
# from fuel.server import recv_arrays, send_arrays
from fuel.converters.ilsvrc2012 import (image_consumer,
                                        load_from_tar,
                                        other_set_producer,
                                        prepare_hdf5_file,
                                        prepare_metadata,
                                        process_train_set,
                                        process_other_set,
                                        read_devkit,
                                        read_metadata_mat_file,
                                        train_set_producer,
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


def test_process_train_set():
    tar_data, names, jpeg_names = create_fake_tar_of_tars(20150925, 5,
                                                          min_num_images=45,
                                                          max_num_images=55)
    all_jpegs = numpy.array(sum(jpeg_names, []))
    numpy.random.RandomState(20150925).shuffle(all_jpegs)
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, len(all_jpegs), 0, 0)
    wnid_map = dict(zip((n.split('.')[0] for n in names), range(len(names))))

    process_train_set(hdf5_file, io.BytesIO(tar_data),
                      len(all_jpegs), wnid_map)

    # Other tests cover that the actual images are what they should be.
    # Just do a basic verification of the filenames and targets.

    assert set(all_jpegs) == set(s.decode('ascii')
                                 for s in hdf5_file['filenames'][:, 0])
    assert len(hdf5_file['encoded_images'][:]) == len(all_jpegs)
    assert len(hdf5_file['targets'][:]) == len(all_jpegs)


def test_process_other_set():
    images, all_filenames = create_fake_jpeg_tar(3, min_num_images=30,
                                                 max_num_images=40,
                                                 gzip_probability=0.0)
    all_filenames_shuffle = numpy.array(all_filenames)
    numpy.random.RandomState(20151202).shuffle(all_filenames_shuffle)
    hdf5_file = MockH5PYFile()
    OFFSET = 50
    prepare_hdf5_file(hdf5_file, OFFSET, len(all_filenames), 0)
    groundtruth = [i % 10 for i in range(len(all_filenames))]
    process_other_set(hdf5_file, 'valid', io.BytesIO(images),
                      groundtruth, OFFSET)

    # Other tests cover that the actual images are what they should be.
    # Just do a basic verification of the filenames.

    assert all(hdf5_file['targets'][OFFSET:, 0] == groundtruth)
    assert all(a.decode('ascii') == b
               for a, b in zip(hdf5_file['filenames'][OFFSET:, 0],
                               all_filenames))


def test_train_set_producer():
    tar_data, names, jpeg_names = create_fake_tar_of_tars(20150923, 5,
                                                          min_num_images=45,
                                                          max_num_images=55)
    all_jpegs = numpy.array(sum(jpeg_names, []))
    numpy.random.RandomState(20150923).shuffle(all_jpegs)
    socket = MockSocket(zmq.PUSH)
    wnid_map = dict(zip((n.split('.')[0] for n in names), range(len(names))))

    train_set_producer(socket, io.BytesIO(tar_data), wnid_map)
    tar_data, names, jpeg_names = create_fake_tar_of_tars(20150923, 5,
                                                          min_num_images=45,
                                                          max_num_images=55)
    for tar_name in names:
        with tarfile.open(fileobj=io.BytesIO(tar_data)) as outer_tar:
            with tarfile.open(fileobj=outer_tar.extractfile(tar_name)) as tar:
                for record in tar:
                    jpeg = record.name
                    metadata_msg = socket.sent.popleft()
                    assert metadata_msg['type'] == 'send_pyobj'
                    assert metadata_msg['flags'] == zmq.SNDMORE
                    key = tar_name.split('.')[0]
                    assert metadata_msg['obj'] == (jpeg, wnid_map[key])

                    image_msg = socket.sent.popleft()
                    assert image_msg['type'] == 'send'
                    assert image_msg['flags'] == 0
                    image_data = load_from_tar(tar, jpeg)
                    assert image_msg['data'] == image_data


def test_image_consumer():
    mock_messages = MOCK_CONSUMER_MESSAGES
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, 4, 5, 8)
    socket = MockSocket(zmq.PULL, to_recv=mock_messages)
    image_consumer(socket, hdf5_file, 4)

    assert_equal(hdf5_file['encoded_images'][0], [6, 6, 6])
    assert_equal(hdf5_file['encoded_images'][1], [1, 8, 1, 2, 0])
    assert_equal(hdf5_file['encoded_images'][2], [1, 9, 7, 9])
    assert_equal(hdf5_file['encoded_images'][3], [1, 8, 6, 7])
    assert_equal(hdf5_file['filenames'][:4], [[b'foo.jpeg'], [b'bar.jpeg'],
                                              [b'baz.jpeg'], [b'bur.jpeg']])
    assert_equal(hdf5_file['targets'][:4], [[2], [3], [5], [7]])


def test_images_consumer_randomized():
    mock_messages = MOCK_CONSUMER_MESSAGES + [
        {'type': 'recv_pyobj', 'flags': zmq.SNDMORE, 'obj': ('jenny.jpeg', 1)},
        {'type': 'recv', 'flags': 0,
         'data': numpy.cast['uint8']([8, 6, 7, 5, 3, 0, 9])}
    ]
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, 4, 5, 8)
    socket = MockSocket(zmq.PULL, to_recv=mock_messages)
    image_consumer(socket, hdf5_file, 5, offset=4, shuffle_seed=0)
    written_data = set(tuple(s) for s in hdf5_file['encoded_images'][4:9])
    expected_data = set(tuple(s['data']) for s in mock_messages[1::2])
    assert written_data == expected_data

    written_targets = set(hdf5_file['targets'][4:9].flatten())
    expected_targets = set(s['obj'][1] for s in mock_messages[::2])
    assert written_targets == expected_targets

    written_filenames = set(hdf5_file['filenames'][4:9].flatten())
    expected_filenames = set(s['obj'][0].encode('ascii')
                             for s in mock_messages[::2])
    assert written_filenames == expected_filenames


def test_other_set_producer():
    # Create some fake data.
    num = 21
    image_archive, filenames = create_fake_jpeg_tar(seed=1979,
                                                    min_num_images=num,
                                                    max_num_images=num)

    groundtruth = numpy.random.RandomState(1979).random_integers(0, 50,
                                                                 size=num)
    assert len(groundtruth) == 21
    gt_lookup = dict(zip(sorted(filenames), groundtruth))
    assert len(gt_lookup) == 21

    def check(which_set):
        # Run other_set_producer and push to a fake socket.
        socket = MockSocket(zmq.PUSH)
        other_set_producer(socket, which_set, io.BytesIO(image_archive),
                           groundtruth)

        # Now verify the data that socket received.
        with tarfile.open(fileobj=io.BytesIO(image_archive)) as tar:
            for im_fn in filenames:
                # Verify the label and flags of the first (metadata)
                # message.
                label = gt_lookup[im_fn]
                metadata_msg = socket.sent.popleft()
                assert metadata_msg['type'] == 'send_pyobj'
                assert metadata_msg['flags'] == zmq.SNDMORE
                assert metadata_msg['obj'] == (im_fn, label)
                # Verify that the second (data) message came from
                # the right place, either a patch file or a TAR.
                data_msg = socket.sent.popleft()
                assert data_msg['type'] == 'send'
                assert data_msg['flags'] == 0
                expected = load_from_tar(tar, im_fn)
                assert data_msg['data'] == expected

    check('valid')
    check('test')


def test_load_from_tar():
    # Setup fake tar files.
    images, all_filenames = create_fake_jpeg_tar(3, min_num_images=200,
                                                 max_num_images=200,
                                                 gzip_probability=0.0)

    with tarfile.open(fileobj=io.BytesIO(images)) as tar:
        for fn in all_filenames:
            image = load_from_tar(tar, fn)
            tar_image = tar.extractfile(fn).read()
            assert image == tar_image


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
