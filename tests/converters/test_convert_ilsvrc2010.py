from collections import deque
import hashlib
import io
import os
import tarfile
from tempfile import NamedTemporaryFile
import gzip

import numpy
from numpy.testing import assert_equal

from PIL import Image
import six
from six.moves import xrange

import zmq

# from fuel.server import recv_arrays, send_arrays
from fuel.converters.ilsvrc2010 import (extract_patch_images,
                                        image_consumer,
                                        load_from_tar_or_patch,
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
                                        TEST_GROUNDTRUTH)
from fuel.utils import find_in_data_path
from tests import skip_if_not_available


class MockSocket(object):
    """Mock of a ZeroMQ PUSH or PULL socket."""
    def __init__(self, socket_type, to_recv=()):
        self.socket_type = socket_type
        if self.socket_type not in (zmq.PUSH, zmq.PULL):
            raise NotImplementedError('only PUSH and PULL currently supported')
        self.sent = deque()
        self.to_recv = deque(to_recv)

    def send(self, data, flags=0, copy=True, track=False):
        assert self.socket_type == zmq.PUSH
        if track:
            # We don't emulate the behaviour required by this flag.
            raise NotImplementedError
        message = {'type': 'send', 'data': data, 'flags': flags, 'copy': copy}
        self.sent.append(message)

    def send_pyobj(self, obj, flags=0, protocol=2):
        assert self.socket_type == zmq.PUSH
        message = {'type': 'send_pyobj', 'obj': obj, 'flags': flags,
                   'protocol': protocol}
        self.sent.append(message)

    def recv(self, flags=0, copy=True, track=False):
        if track:
            # We don't emulate the behaviour required by this flag.
            raise NotImplementedError
        message = self.to_recv.popleft()
        assert message['type'] == 'recv'
        if 'flags' in message:
            assert message['flags'] == flags, 'flags did not match expected'
        if 'copy' in message:
            assert message['copy'] == copy, 'copy did not match expected'
        return message['data']

    def recv_pyobj(self, flags=0):
        message = self.to_recv.popleft()
        assert message['type'] == 'recv_pyobj'
        if 'flags' in message:
            assert flags == message['flags']
        return message['obj']


class MockH5PYData(object):
    def __init__(self, shape, dtype):
        self.data = numpy.empty(shape, dtype)
        self.dims = MockH5PYDims(len(shape))

    def __setitem__(self, where, what):
        self.data[where] = what

    def __getitem__(self, index):
        return self.data[index]

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype


class MockH5PYFile(dict):
    filename = 'NOT_A_REAL_FILE.hdf5'

    def __init__(self):
        self.attrs = {}
        self.flushed = 0
        self.opened = False
        self.closed = False

    def create_dataset(self, name, shape, dtype):
        self[name] = MockH5PYData(shape, dtype)

    def flush(self):
        self.flushed += 1

    def __enter__(self):
        self.opened = True

    def __exit__(self, type, value, traceback):
        self.closed = True


class MockH5PYDim(object):
    def __init__(self, dims):
        self.dims = dims
        self.scales = []

    def attach_scale(self, dataset):
        # I think this is necessary for it to be valid?
        assert dataset in self.dims.scales.values()
        self.scales.append(dataset)


class MockH5PYDims(object):
    def __init__(self, ndim):
        self._dims = [MockH5PYDim(self) for _ in xrange(ndim)]
        self.scales = {}

    def create_scale(self, dataset, name):
        self.scales[name] = dataset

    def __getitem__(self, index):
        return self._dims[index]


def create_jpeg_data(image):
    """Create a JPEG in memory.

    Parameters
    ----------
    image : ndarray, 3-dimensional
        Array data representing the image to save. Mode ('L', 'RGB',
        'CMYK') will be determined from the last (third) axis.

    Returns
    -------
    jpeg_data : bytes
        The image encoded as a JPEG, returned as raw bytes.

    """
    if image.shape[-1] == 1:
        mode = 'L'
    elif image.shape[-1] == 3:
        mode = 'RGB'
    elif image.shape[-1] == 4:
        mode = 'CMYK'
    else:
        raise ValueError("invalid shape")
    pil_image = Image.frombytes(mode=mode, size=image.shape[:2],
                                data=image.tobytes())
    jpeg_data = io.BytesIO()
    pil_image.save(jpeg_data, format='JPEG')
    return jpeg_data.getvalue()


def create_fake_jpeg_tar(seed, min_num_images=5, max_num_images=50,
                         min_size=20, size_range=30, filenames=None,
                         random=True, gzip_probability=0.5, offset=0):
    """Create a TAR file of randomly generated JPEG files.

    Parameters
    ----------
    seed : int or sequence
        Seed for a `numpy.random.RandomState`.
    min_num_images : int, optional
        The minimum number of images to put in the TAR file.
    max_num_images : int, optional
        The maximum number of images to put in the TAR file.
    min_size : int, optional
        The minimum width and minimum height of each image.
    size_range : int, optional
        Maximum number of pixels added to `min_size` for image
        dimensions.
    filenames : list, optional
        If provided, use these filenames. Otherwise generate them
        randomly. Must be at least `max_num_images` long.
    random : bool, optional
        If `False`, substitute an image full of a single number,
        the order of that image in processing.
    gzip_probability : float, optional
        With this probability, randomly gzip the JPEG file without
        appending a gzip suffix.
    offset : int, optional
        Where to start the hashes for filenames. Default: 0.

    Returns
    -------
    tar_data : bytes
        A TAR file represented as raw bytes, containing between
        `min_num_images` and `max_num_images` JPEG files (inclusive).

    Notes
    -----
    Randomly choose between RGB, L and CMYK mode images. Also randomly
    gzips JPEGs to simulate the idiotic distribution format of
    ILSVRC2010.

    """
    rng = numpy.random.RandomState(seed)
    images = []
    if filenames is None:
        files = []
    else:
        if len(filenames) < max_num_images:
            raise ValueError('need at least max_num_images = %d filenames' %
                             max_num_images)
        files = filenames
    for i in xrange(rng.random_integers(min_num_images, max_num_images)):
        if filenames is None:
            max_len = 27  # so that with suffix, 32 characters
            files.append('%s.JPEG' %
                         hashlib.sha1(bytes(i + offset)).hexdigest()[:max_len])
        im = rng.random_integers(0, 255,
                                 size=(rng.random_integers(min_size,
                                                           min_size +
                                                           size_range),
                                       rng.random_integers(min_size,
                                                           min_size +
                                                           size_range),
                                       rng.random_integers(1, 4)))
        if not random:
            im *= 0
            assert (im == 0).all()
            im += i
            assert numpy.isscalar(i)
            assert (im == i).all()
        if im.shape[-1] == 2:
            im = im[:, :, :1]
        images.append(im)
    files = sorted(files)
    temp_tar = io.BytesIO()
    with tarfile.open(fileobj=temp_tar, mode='w') as tar:
        for fn, image in zip(files, images):
            try:
                with NamedTemporaryFile(mode='wb', suffix='.JPEG',
                                        delete=False) as f:
                    if rng.uniform() < gzip_probability:
                        gzip_data = io.BytesIO()
                        with gzip.GzipFile(mode='wb', fileobj=gzip_data) as gz:
                            gz.write(create_jpeg_data(image))
                        f.write(gzip_data.getvalue())
                    else:
                        f.write(create_jpeg_data(image))
                tar.add(f.name, arcname=fn)
            finally:
                os.remove(f.name)
    ordered_files = []
    with tarfile.open(fileobj=io.BytesIO(temp_tar.getvalue()),
                      mode='r') as tar:
        for info in tar.getmembers():
            ordered_files.append(info.name)
    return temp_tar.getvalue(), ordered_files


def create_fake_tar_of_tars(seed, num_inner_tars, *args, **kwargs):
    """Create a nested TAR of TARs of JPEGs.

    Parameters
    ----------
    seed : int or sequence
        Seed for a `numpy.random.RandomState`.
    num_inner_tars : int
        Number of TAR files to place inside.

    Returns
    -------
    tar_data : bytes
        A TAR file represented as raw bytes, TAR files of generated
        JPEGs.
    names : list
        Names of the inner TAR files.
    jpeg_names : list of lists
        A list of lists containing the names of JPEGs in each inner TAR.


    Notes
    -----
    Remainder of positional and keyword arguments are passed on to
    :func:`create_fake_jpeg_tars`.

    """
    seeds = numpy.arange(num_inner_tars) + seed
    tars, fns = [], []
    offset = 0
    for s in seeds:
        tar, fn = create_fake_jpeg_tar(s, *args, offset=offset, **kwargs)
        tars.append(tar)
        fns.append(fn)
        offset += len(fn)
    names = sorted(str(abs(hash(str(-i - 1)))) + '.tar'
                   for i, t in enumerate(tars))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as outer:
        for tar, name in zip(tars, names):
            try:
                with NamedTemporaryFile(mode='wb', suffix='.tar',
                                        delete=False) as f:
                    f.write(tar)
                outer.add(f.name, arcname=name)
            finally:
                os.remove(f.name)
    return data.getvalue(), names, fns


def create_fake_patch_images(filenames=None, num_train=14, num_valid=15,
                             num_test=21):
    if filenames is None:
        num = num_train + num_valid + num_test
        filenames = ['%x' % abs(hash(str(i))) + '.JPEG' for i in xrange(num)]
    else:
        filenames = list(filenames)  # Copy, so list not modified in-place.
    filenames[:num_train] = ['train/' + f
                             for f in filenames[:num_train]]
    filenames[num_train:num_train + num_valid] = [
        'val/' + f for f in filenames[num_train:num_train + num_valid]
    ]
    filenames[num_train + num_valid:] = [
        'test/' + f for f in filenames[num_train + num_valid:]
    ]
    tar = create_fake_jpeg_tar(1, min_num_images=len(filenames),
                               max_num_images=len(filenames),
                               filenames=filenames, random=False,
                               gzip_probability=0.0)[0]
    return tar


def test_prepare_metadata():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE, TEST_GROUNDTRUTH])
    devkit_path = find_in_data_path(DEVKIT_ARCHIVE)
    test_gt_path = find_in_data_path(TEST_GROUNDTRUTH)
    n_train, v_gt, t_gt, wnid_map = prepare_metadata(devkit_path,
                                                     test_gt_path)
    assert n_train == 1261406
    assert len(v_gt) == 50000
    assert len(t_gt) == 150000
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
                     for r in rows])

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
    assert set(test_splits.keys()) == set([u'encoded_images', u'targets',
                                           u'filenames'])

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
    assert hdf5_file['targets'].shape[0] == 17
    assert hdf5_file['targets'].shape[1] == 1
    assert len(hdf5_file['targets'].shape) == 2
    assert hdf5_file['targets'].dtype == dtype('int16')


def test_process_train_set():
    tar_data, names, jpeg_names = create_fake_tar_of_tars(20150925, 5,
                                                          min_num_images=45,
                                                          max_num_images=55)
    all_jpegs = numpy.array(sum(jpeg_names, []))
    numpy.random.RandomState(20150925).shuffle(all_jpegs)
    patched_files = all_jpegs[:10]
    patches_data = create_fake_patch_images(filenames=patched_files,
                                            num_train=10, num_valid=0,
                                            num_test=0)
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, len(all_jpegs), 0, 0)
    wnid_map = dict(zip((n.split('.')[0] for n in names), range(len(names))))

    process_train_set(hdf5_file, io.BytesIO(tar_data),
                      io.BytesIO(patches_data), len(all_jpegs),
                      wnid_map)

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
    patched_files = all_filenames_shuffle[:15]
    patches_data = create_fake_patch_images(filenames=patched_files,
                                            num_train=0, num_valid=15,
                                            num_test=0)
    hdf5_file = MockH5PYFile()
    OFFSET = 50
    prepare_hdf5_file(hdf5_file, OFFSET, len(all_filenames), 0)
    groundtruth = [i % 10 for i in range(len(all_filenames))]
    process_other_set(hdf5_file, 'valid', io.BytesIO(images),
                      io.BytesIO(patches_data), groundtruth, OFFSET)

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
    patched_files = all_jpegs[:10]
    patches_data = create_fake_patch_images(filenames=patched_files,
                                            num_train=10, num_valid=0,
                                            num_test=0)
    train_patches = extract_patch_images(io.BytesIO(patches_data), 'train')
    socket = MockSocket(zmq.PUSH)
    wnid_map = dict(zip((n.split('.')[0] for n in names), range(len(names))))

    train_set_producer(socket, io.BytesIO(tar_data), io.BytesIO(patches_data),
                       wnid_map)
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
                    if jpeg in train_patches:
                        assert image_msg['data'] == train_patches[jpeg]
                    else:
                        image_data, _ = load_from_tar_or_patch(tar, jpeg,
                                                               train_patches)
                        assert image_msg['data'] == image_data


MOCK_CONSUMER_MESSAGES = [
    {'type': 'recv_pyobj', 'flags': zmq.SNDMORE, 'obj': ('foo.jpeg', 2)},
    {'type': 'recv', 'flags': 0, 'data': numpy.cast['uint8']([6, 6, 6])},
    {'type': 'recv_pyobj', 'flags': zmq.SNDMORE, 'obj': ('bar.jpeg', 3)},
    {'type': 'recv', 'flags': 0, 'data': numpy.cast['uint8']([1, 8, 1, 2, 0])},
    {'type': 'recv_pyobj', 'flags': zmq.SNDMORE, 'obj': ('baz.jpeg', 5)},
    {'type': 'recv', 'flags': 0, 'data': numpy.cast['uint8']([1, 9, 7, 9])},
    {'type': 'recv_pyobj', 'flags': zmq.SNDMORE, 'obj': ('bur.jpeg', 7)},
    {'type': 'recv', 'flags': 0, 'data': numpy.cast['uint8']([1, 8, 6, 7])},
]


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
    patches = create_fake_patch_images(filenames=filenames,
                                       num_train=7, num_valid=7, num_test=7)

    valid_patches = extract_patch_images(io.BytesIO(patches), 'valid')
    test_patches = extract_patch_images(io.BytesIO(patches), 'test')
    assert len(valid_patches) == 7
    assert len(test_patches) == 7

    groundtruth = numpy.random.RandomState(1979).random_integers(0, 50,
                                                                 size=num)
    assert len(groundtruth) == 21
    gt_lookup = dict(zip(sorted(filenames), groundtruth))
    assert len(gt_lookup) == 21

    def check(which_set, set_patches):
        # Run other_set_producer and push to a fake socket.
        socket = MockSocket(zmq.PUSH)
        other_set_producer(socket, which_set, io.BytesIO(image_archive),
                           io.BytesIO(patches), groundtruth)

        # Now verify the data that socket received.
        with tarfile.open(fileobj=io.BytesIO(image_archive)) as tar:
            num_patched = 0
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
                expected, patched = load_from_tar_or_patch(tar, im_fn,
                                                           set_patches)
                num_patched += int(patched)
                assert data_msg['data'] == expected
            assert num_patched == len(set_patches)

    check('valid', valid_patches)
    check('test', test_patches)


def test_load_from_tar_or_patch():
    # Setup fake tar files.
    images, all_filenames = create_fake_jpeg_tar(3, min_num_images=200,
                                                 max_num_images=200,
                                                 gzip_probability=0.0)
    patch_data = create_fake_patch_images(all_filenames[::4], num_train=50,
                                          num_valid=0, num_test=0)

    patches = extract_patch_images(io.BytesIO(patch_data), 'train')

    assert len(patches) == 50
    with tarfile.open(fileobj=io.BytesIO(images)) as tar:
        for fn in all_filenames:
            image, patched = load_from_tar_or_patch(tar, fn, patches)
            if fn in patches:
                assert image == patches[fn]
                assert patched
            else:
                tar_image = tar.extractfile(fn).read()
                assert image == tar_image
                assert not patched


def test_read_devkit():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE])
    synsets, cost_mat, raw_valid_gt = read_devkit(
        find_in_data_path(DEVKIT_ARCHIVE))
    # synset and cost_matrix sanity tests appear in test_read_metadata_mat_file
    assert raw_valid_gt.min() == 1
    assert raw_valid_gt.max() == 1000
    assert raw_valid_gt.dtype.kind == 'i'
    assert raw_valid_gt.shape == (50000,)


def test_read_metadata_mat_file():
    skip_if_not_available(datasets=[DEVKIT_ARCHIVE])
    with tarfile.open(find_in_data_path(DEVKIT_ARCHIVE)) as tar:
        meta_mat = tar.extractfile(DEVKIT_META_PATH)
        synsets, cost_mat = read_metadata_mat_file(meta_mat)
    assert (synsets['ILSVRC2010_ID'] ==
            numpy.arange(1, len(synsets) + 1)).all()
    assert synsets['num_train_images'][1000:].sum() == 0
    assert (synsets['num_train_images'][:1000] > 0).all()
    assert synsets.ndim == 1
    assert synsets['wordnet_height'].min() == 0
    assert synsets['wordnet_height'].max() == 19
    assert synsets['WNID'].dtype == numpy.dtype('S9')
    assert (synsets['num_children'][:1000] == 0).all()
    assert (synsets['children'][:1000] == -1).all()

    # Assert the basics about the cost matrix.
    assert cost_mat.shape == (1000, 1000)
    assert cost_mat.dtype == 'uint8'
    assert cost_mat.min() == 0
    assert cost_mat.max() == 18
    assert (cost_mat == cost_mat.T).all()
    # Assert that the diagonal is 0.
    assert (cost_mat.flat[::1001] == 0).all()


def test_extract_patch_images():
    tar = create_fake_patch_images()
    assert len(extract_patch_images(io.BytesIO(tar), 'train')) == 14
    assert len(extract_patch_images(io.BytesIO(tar), 'valid')) == 15
    assert len(extract_patch_images(io.BytesIO(tar), 'test')) == 21
