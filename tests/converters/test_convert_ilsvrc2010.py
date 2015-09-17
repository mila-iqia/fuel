from collections import deque
import io
import os
import tarfile
from tempfile import NamedTemporaryFile
import unittest
import gzip

import numpy
from PIL import Image
from six.moves import xrange
import zmq

# from fuel.server import recv_arrays, send_arrays
from fuel.datasets import H5PYDataset
from fuel.converters.ilsvrc2010 import (extract_patch_images,
                                        load_from_tar_or_patch,
                                        read_devkit,
                                        prepare_hdf5_file)
from tests import skip_if_not_available

class MockSocket(object):
    """Mock of a ZeroMQ PUSH or PULL socket."""
    def __init__(self, socket_type, to_recv):
        self.socket_type = socket_type
        if self.socket_type not in (zmq.PUSH, zmq.PULL):
            raise NotImplementedError('only PUSH and PULL currently supported')
        self.sent = deque()
        self.to_recv = deque(to_recv)

    def send(data, flags=0, copy=True, track=False):
        assert self.socket_type == zmq.PUSH
        if track:
            # We don't emulate the behaviour required by this flag.
            raise NotImplementedError
        message = {'type': 'send', 'data': data, 'flags': flags, 'copy': copy}
        self.sent.append(message)

    def send_pyobj(obj, flags=0, protocol=2):
        assert self.socket_type == zmq.PUSH
        message = {'type': 'send_pyobj', 'obj': obj, 'flags': flags,
                   'protocol': protocol}
        self.sent.append(message)

    def recv(flags=0, copy=True, track=False):
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

    def recv_pyobj(flags=0):
        message = self.to_recv.popleft()
        assert message['type'] == 'recv_pyobj'
        if 'flags' in message:
            assert flags == message['flags']
        return message['obj']


class MockH5PYData(object):
    def __init__(self, shape, dtype):
        self.data = numpy.empty(shape, dtype)
        self.dims = MockH5PYDims(len(shape))
        self.written = 0

    def __setitem__(self, where, what):
        self.data[where] = what
        self.written += len(what)

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
                         random=True, gzip_probability=0.2):
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
    gzip_probability : float
        With this probability, randomly gzip the JPEG file without
        appending a gzip suffix.

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
            files.append('%x.JPEG' % abs(hash(str(i))))
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
    return temp_tar.getvalue(), files[:len(images)]


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
    rng = numpy.random.RandomState(seed)
    seeds = rng.random_integers(0, 500000, size=(num_inner_tars,))
    tars, fns = list(zip(*[create_fake_jpeg_tar(s, *args, **kwargs)
                           for s in seeds]))
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
        filenames = ['%x' % abs(hash(str(i))) + '.JPEG' for i in xrange(50)]
    assert num_train + num_valid + num_test == len(filenames)
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
    return tar, filenames


def push_pull_socket_pair(context):
    pull = context.socket(zmq.PULL)
    pull_port = pull.bind_to_random_port('tcp://*')
    push = context.socket(zmq.PUSH)
    push.connect('tcp://localhost:{}'.format(pull_port))
    return push, pull


def test_prepare_metadata():
    raise unittest.SkipTest("TODO")


def test_prepare_hdf5_file():
    hdf5_file = MockH5PYFile()
    prepare_hdf5_file(hdf5_file, 10, 5, 2)

    train_splits = H5PYDataset.get_start_stop(hdf5_file, 'train')
    assert all(v == (0, 10) for v in train_splits.values())
    assert train_splits.keys() == set(['encoded_images', 'targets',
                                       'filenames'])

    valid_splits = H5PYDataset.get_start_stop(hdf5_file, 'valid')
    assert all(v == (10, 15) for v in valid_splits.values())
    assert valid_splits.keys() == set(['encoded_images', 'targets',
                                       'filenames'])

    test_splits = H5PYDataset.get_start_stop(hdf5_file, 'test')
    assert all(v == (15, 17) for v in test_splits.values())
    assert test_splits.keys() == set(['encoded_images', 'targets',
                                      'filenames'])

    from numpy import dtype

    assert hdf5_file['encoded_images'].shape[0] == 17
    assert len(hdf5_file['encoded_images'].shape) == 1
    assert hdf5_file['encoded_images'].dtype.kind == 'O'
    assert hdf5_file['encoded_images'].dtype.metadata['vlen'] == dtype('uint8')

    assert hdf5_file['filenames'].shape[0] == 17
    assert len(hdf5_file['filenames'].shape) == 2
    assert hdf5_file['filenames'].dtype == dtype('S32')

    assert hdf5_file['targets'].shape[0] == 17
    assert hdf5_file['targets'].shape[1] == 1
    assert len(hdf5_file['targets'].shape) == 2
    assert hdf5_file['targets'].dtype == dtype('int16')


def test_process_train_set():
    raise unittest.SkipTest("TODO")


def test_process_other_set():
    raise unittest.SkipTest("TODO")


def test_train_set_producer():
    raise unittest.SkipTest("TODO")


def test_image_consumer():
    raise unittest.SkipTest("TODO")


def test_other_set_producer():
    raise unittest.SkipTest("TODO")


def test_load_from_tar_or_patch():
    images, all_filenames = create_fake_jpeg_tar(3, min_num_images=200,
                                                 max_num_images=200,
                                                 gzip_probability=0.0)
    patch_data, _ = create_fake_patch_images(all_filenames[::4], num_train=50,
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
    devkit_filename = 'ILSVRC2010_devkit-1.0.tar.gz'
    skip_if_not_available(datasets=[devkit_filename])
    synsets, cost_mat, raw_valid_gt = read_devkit(
        os.path.join(config.data_path, devkit_filename))
    assert (synsets['ILSVRC2010_ID'] ==
            numpy.arange(1, len(synsets) + 1)).all()
    assert synsets['num_train_images'][1000:].sum() == 0
    assert (synsets['num_train_images'][:1000] > 0).all()
    assert synsets.ndim == 1
    assert cost_mat.shape == (1000, 1000)
    assert cost_mat.dtype == 'uint8'
    assert (cost_mat.flat[::1001] == 0).all()


def test_read_metadata_mat_file():
    raise unittest.SkipTest("TODO")


def test_extract_patch_images():
    tar, _ = create_fake_patch_images()
    assert len(extract_patch_images(io.BytesIO(tar), 'train')) == 14
    assert len(extract_patch_images(io.BytesIO(tar), 'valid')) == 15
    assert len(extract_patch_images(io.BytesIO(tar), 'test')) == 21
