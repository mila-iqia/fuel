from __future__ import division
from collections import OrderedDict
from functools import partial
import gzip
import io
import os
import logging
import os.path

import h5py
import numpy
from picklable_itertools.extras import equizip
from PIL import Image
from scipy.io.matlab import loadmat
from six.moves import zip, xrange
import zmq

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets import H5PYDataset
from fuel.utils.formats import tar_open
from fuel.utils.parallel import producer_consumer
from fuel import config

log = logging.getLogger(__name__)

DEVKIT_ARCHIVE = 'ILSVRC2012_devkit_t12.tar.gz'
DEVKIT_META_PATH = 'ILSVRC2012_devkit_t12/data/meta.mat'
DEVKIT_VALID_GROUNDTRUTH_PATH = ('ILSVRC2012_devkit_t12/data/'
                                 'ILSVRC2012_validation_ground_truth.txt')
TRAIN_IMAGES_TAR = 'ILSVRC2012_img_train.tar'
VALID_IMAGES_TAR = 'ILSVRC2012_img_val.tar'
TEST_IMAGES_TAR = 'ILSVRC2012_img_test.tar'
IMAGE_TARS = (TRAIN_IMAGES_TAR, VALID_IMAGES_TAR, TEST_IMAGES_TAR)
ALL_FILES = (DEVKIT_ARCHIVE, ) + IMAGE_TARS


@check_exists(required_files=ALL_FILES)
def convert_ilsvrc2012(directory, output_directory,
                       output_filename='ilsvrc2012.hdf5',
                       shuffle_seed=config.default_seed):
    """Converter for data from the ILSVRC 2012 competition.

    Source files for this dataset can be obtained by registering at
    [ILSVRC2012WEB].

    Parameters
    ----------
    input_directory : str
        Path from which to read raw data files.
    output_directory : str
        Path to which to save the HDF5 file.
    output_filename : str, optional
        The output filename for the HDF5 file. Default: 'ilsvrc2012.hdf5'.
    shuffle_seed : int or sequence, optional
        Seed for a random number generator used to shuffle the order
        of the training set on disk, so that sequential reads will not
        be ordered by class.

    .. [ILSVRC2012WEB] http://image-net.org/challenges/LSVRC/2012/index

    """
    devkit_path = os.path.join(directory, DEVKIT_ARCHIVE)
    train, valid, test = [os.path.join(directory, fn) for fn in IMAGE_TARS]
    n_train, valid_groundtruth, n_test, wnid_map = prepare_metadata(devkit_path)
    n_valid = len(valid_groundtruth)
    output_path = os.path.join(output_directory, output_filename)

    with h5py.File(output_path, 'w') as f:
        log.info('Creating HDF5 datasets...')
        prepare_hdf5_file(f, n_train, n_valid, n_test)
        log.info('Processing training set...')
        process_train_set(f, train, n_train, wnid_map, shuffle_seed)
        log.info('Processing validation set...')
        process_other_set(f, 'valid', valid, valid_groundtruth, n_train)
        log.info('Processing test set...')
        process_other_set(f, 'test', test, (None, ) * n_test, n_train + n_valid)
        log.info('Done.')

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the ILSVRC2012 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `ilsvrc2012` command.

    """
    subparser.add_argument(
        "--shuffle-seed", help="Seed to use for randomizing order of the "
                               "training set on disk.",
        default=config.default_seed, type=int, required=False)
    return convert_ilsvrc2012


def prepare_metadata(devkit_archive):
    """Extract dataset metadata required for HDF5 file setup.

    Parameters
    ----------
    devkit_archive : str or file-like object
        The filename or file-handle for the gzipped TAR archive
        containing the ILSVRC2012 development kit.

    Returns
    -------
    n_train : int
        The number of examples in the training set.
    valid_groundtruth : ndarray, 1-dimensional
        An ndarray containing the validation set groundtruth in terms of
        0-based class indices.
    n_test : int
        The number of examples in the test set
    wnid_map : dict
        A dictionary that maps WordNet IDs to 0-based class indices.

    """
    # Read what's necessary from the development kit.
    synsets, raw_valid_groundtruth = read_devkit(devkit_archive)

    # Mapping to take WordNet IDs to our internal 0-999 encoding.
    wnid_map = dict(zip((s.decode('utf8') for s in synsets['WNID']),
                        xrange(1000)))

    # Map the 'ILSVRC2012 ID' to our zero-based ID.
    ilsvrc_id_to_zero_based = dict(zip(synsets['ILSVRC2012_ID'],
                                   xrange(len(synsets))))

    # Map the validation set groundtruth to 0-999 labels.
    valid_groundtruth = [ilsvrc_id_to_zero_based[id_]
                         for id_ in raw_valid_groundtruth]

    # Get number of test examples from the test archive
    with tar_open(TEST_IMAGES_TAR) as f:
        n_test = sum(1 for _ in f)

    # Ascertain the number of filenames to prepare appropriate sized
    # arrays.
    n_train = int(synsets['num_train_images'].sum())
    log.info('Training set: {} images'.format(n_train))
    log.info('Validation set: {} images'.format(len(valid_groundtruth)))
    log.info('Test set: {} images'.format(n_test))
    n_total = n_train + len(valid_groundtruth) + n_test
    log.info('Total (train/valid): {} images'.format(n_total))
    return n_train, valid_groundtruth, n_test, wnid_map


def create_splits(n_train, n_valid, n_test):
    n_total = n_train + n_valid + n_test
    tuples = {}
    tuples['train'] = (0, n_train)
    tuples['valid'] = (n_train, n_train + n_valid)
    tuples['test'] = (n_train + n_valid, n_total)
    sources = ['encoded_images', 'targets', 'filenames']
    return OrderedDict(
        (split, OrderedDict((source, tuples[split]) for source in sources
                            if source != 'targets' or split != 'test'))
        for split in ('train', 'valid', 'test')
    )


def prepare_hdf5_file(hdf5_file, n_train, n_valid, n_test):
    """Create datasets within a given HDF5 file.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write.
    n_train : int
        The number of training set examples.
    n_valid : int
        The number of validation set examples.
    n_test : int
        The number of test set examples.

    """
    n_total = n_train + n_valid + n_test
    n_labeled = n_train + n_valid
    splits = create_splits(n_train, n_valid, n_test)
    hdf5_file.attrs['split'] = H5PYDataset.create_split_array(splits)
    vlen_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    hdf5_file.create_dataset('encoded_images', shape=(n_total,),
                             dtype=vlen_dtype)
    hdf5_file.create_dataset('targets', shape=(n_labeled, 1), dtype=numpy.int16)
    hdf5_file.create_dataset('filenames', shape=(n_total, 1), dtype='S32')


def process_train_set(hdf5_file, train_archive, n_train,
                      wnid_map, shuffle_seed=None):
    """Process the ILSVRC2012 training set.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write. Assumes `features`, `targets`
        and `filenames` already exist and have first dimension larger than
        `n_train`.
    train_archive :  str or file-like object
        Filename or file handle for the TAR archive of training images.
    n_train : int
        The number of items in the training set.
    wnid_map : dict
        A dictionary mapping WordNet IDs to class indices.
    shuffle_seed : int or sequence, optional
        Seed for a NumPy random number generator that permutes the
        training set on disk. If `None`, no permutation is performed
        (this is the default).

    """
    producer = partial(train_set_producer, train_archive=train_archive,
                       wnid_map=wnid_map)
    consumer = partial(image_consumer, hdf5_file=hdf5_file,
                       num_expected=n_train, shuffle_seed=shuffle_seed)
    producer_consumer(producer, consumer)


def _write_to_hdf5(hdf5_file, index, image_filename, image_data,
                   class_index):
    hdf5_file['filenames'][index] = image_filename.encode('ascii')
    hdf5_file['encoded_images'][index] = image_data
    if class_index is not None:
        hdf5_file['targets'][index] = class_index


def train_set_producer(socket, train_archive, wnid_map):
    """Load/send images from the training set TAR file or patch images.

    Parameters
    ----------
    socket : :class:`zmq.Socket`
        PUSH socket on which to send loaded images.
    train_archive :  str or file-like object
        Filename or file handle for the TAR archive of training images.
    wnid_map : dict
        A dictionary that maps WordNet IDs to 0-based class indices.
        Used to decode the filenames of the inner TAR files.

    """
    with tar_open(train_archive) as tar:
        for inner_tar_info in tar:
            with tar_open(tar.extractfile(inner_tar_info.name)) as inner:
                wnid = inner_tar_info.name.split('.')[0]
                class_index = wnid_map[wnid]
                filenames = sorted(info.name for info in inner
                                   if info.isfile())
                images_gen = (load_from_tar(inner, filename)
                              for filename in filenames)
                stream = equizip(filenames, images_gen)
                for image_fn, image_data in stream:
                    socket.send_pyobj((image_fn, class_index), zmq.SNDMORE)
                    socket.send(image_data)


def image_consumer(socket, hdf5_file, num_expected, shuffle_seed=None,
                   offset=0):
    """Fill an HDF5 file with incoming images from a socket.

    Parameters
    ----------
    socket : :class:`zmq.Socket`
        PULL socket on which to receive images.
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write. Assumes `features`, `targets`
        and `filenames` already exist and have first dimension larger than
        `sum(images_per_class)`.
    num_expected : int
        The number of items we expect to be sent over the socket.
    shuffle_seed : int or sequence, optional
        Seed for a NumPy random number generator that permutes the
        images on disk.
    offset : int, optional
        The offset in the HDF5 datasets at which to start writing
        received examples. Defaults to 0.

    """
    with progress_bar('images', maxval=num_expected) as pb:
        if shuffle_seed is None:
            index_gen = iter(xrange(num_expected))
        else:
            rng = numpy.random.RandomState(shuffle_seed)
            index_gen = iter(rng.permutation(num_expected))
        for i, num in enumerate(index_gen):
            image_filename, class_index = socket.recv_pyobj(zmq.SNDMORE)
            image_data = numpy.fromstring(socket.recv(), dtype='uint8')
            _write_to_hdf5(hdf5_file, num + offset, image_filename,
                           image_data, class_index)
            pb.update(i + 1)


def process_other_set(hdf5_file, which_set, image_archive,
                      groundtruth, offset):
    """Process the validation or test set.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write. Assumes `features`, `targets`
        and `filenames` already exist and have first dimension larger than
        `sum(images_per_class)`.
    which_set : str
        Which set of images is being processed. One of 'train', 'valid',
        'test'.  Used for extracting the appropriate images from the patch
        archive.
    image_archive : str or file-like object
        The filename or file-handle for the TAR archive containing images.
    groundtruth : iterable
        Iterable container containing scalar 0-based class index for each
        image, sorted by filename.
    offset : int
        The offset in the HDF5 datasets at which to start writing.

    """
    producer = partial(other_set_producer, image_archive=image_archive,
                       groundtruth=groundtruth, which_set=which_set)
    consumer = partial(image_consumer, hdf5_file=hdf5_file,
                       num_expected=len(groundtruth), offset=offset)
    producer_consumer(producer, consumer)


def other_set_producer(socket, which_set, image_archive, groundtruth):
    """Push image files read from the valid/test set TAR to a socket.

    Parameters
    ----------
    socket : :class:`zmq.Socket`
        PUSH socket on which to send images.
    which_set : str
        Which set of images is being processed. One of 'train', 'valid',
        'test'.  Used for extracting the appropriate images from the patch
        archive.
    image_archive : str or file-like object
        The filename or file-handle for the TAR archive containing images.
    groundtruth : iterable
        Iterable container containing scalar 0-based class index for each
        image, sorted by filename.

    """
    with tar_open(image_archive) as tar:
        filenames = sorted(info.name for info in tar if info.isfile())
        images = (load_from_tar(tar, filename)
                  for filename in filenames)
        stream = equizip(filenames, images, groundtruth)
        for image_fn, image_data, class_index in stream:
            socket.send_pyobj((image_fn, class_index), zmq.SNDMORE)
            socket.send(image_data, copy=False)


def load_from_tar(tar, image_filename):
    """Do everything necessary to process an image inside a TAR.

    Parameters
    ----------
    tar : `TarFile` instance
        The tar from which to read `image_filename`.
    image_filename : str
        Fully-qualified path inside of `tar` from which to read an
        image file.

    Returns
    -------
    image_data : bytes
        The JPEG bytes representing the image from the TAR archive.

    """
    try:
        image_bytes = tar.extractfile(image_filename).read()
        numpy.array(Image.open(io.BytesIO(image_bytes)))
    except (IOError, OSError):
        with gzip.GzipFile(fileobj=tar.extractfile(image_filename)) as gz:
            image_bytes = gz.read()
            numpy.array(Image.open(io.BytesIO(image_bytes)))
    return image_bytes


def read_devkit(f):
    """Read relevant information from the development kit archive.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle for the gzipped TAR archive
        containing the ILSVRC2012 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        See :func:`read_metadata_mat_file` for details.
    raw_valid_groundtruth : ndarray, 1-dimensional, int16
        The labels for the ILSVRC2012 validation set,
        distributed with the development kit code.

    """
    with tar_open(f) as tar:
        # Metadata table containing class hierarchy, textual descriptions, etc.
        meta_mat = tar.extractfile(DEVKIT_META_PATH)
        synsets = read_metadata_mat_file(meta_mat)

        # Raw validation data groundtruth, ILSVRC2012 IDs. Confusingly
        # distributed inside the development kit archive.
        raw_valid_groundtruth = numpy.loadtxt(tar.extractfile(
            DEVKIT_VALID_GROUNDTRUTH_PATH), dtype=numpy.int16)
    return synsets, raw_valid_groundtruth


def read_metadata_mat_file(meta_mat):
    """Read ILSVRC2012 metadata from the distributed MAT file.

    Parameters
    ----------
    meta_mat : str or file-like object
        The filename or file-handle for `meta.mat` from the
        ILSVRC2012 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        A table containing ILSVRC2012 metadata for the "synonym sets"
        or "synsets" that comprise the classes and superclasses,
        including the following fields:
         * `ILSVRC2012_ID`: the integer ID used in the original
           competition data.
         * `WNID`: A string identifier that uniquely identifies
           a synset in ImageNet and WordNet.
         * `wordnet_height`: The length of the longest path to
           a leaf node in the FULL ImageNet/WordNet hierarchy
           (leaf nodes in the FULL ImageNet/WordNet hierarchy
           have `wordnet_height` 0).
         * `gloss`: A string representation of an English
           textual description of the concept represented by
           this synset.
         * `num_children`: The number of children in the hierarchy
           for this synset.
         * `words`: A string representation, comma separated,
           of different synoym words or phrases for the concept
           represented by this synset.
         * `children`: A vector of `ILSVRC2012_ID`s of children
           of this synset, padded with -1. Note that these refer
           to `ILSVRC2012_ID`s from the original data and *not*
           the zero-based index in the table.
         * `num_train_images`: The number of training images for
           this synset.

    """
    mat = loadmat(meta_mat, squeeze_me=True)
    synsets = mat['synsets']
    new_dtype = numpy.dtype([
        ('ILSVRC2012_ID', numpy.int16),
        ('WNID', ('S', max(map(len, synsets['WNID'])))),
        ('wordnet_height', numpy.int8),
        ('gloss', ('S', max(map(len, synsets['gloss'])))),
        ('num_children', numpy.int8),
        ('words', ('S', max(map(len, synsets['words'])))),
        ('children', (numpy.int8, max(synsets['num_children']))),
        ('num_train_images', numpy.uint16)
    ])
    new_synsets = numpy.empty(synsets.shape, dtype=new_dtype)
    for attr in ['ILSVRC2012_ID', 'WNID', 'wordnet_height', 'gloss',
                 'num_children', 'words', 'num_train_images']:
        new_synsets[attr] = synsets[attr]
    children = [numpy.atleast_1d(ch) for ch in synsets['children']]
    padded_children = [
        numpy.concatenate((c,
                           -numpy.ones(new_dtype['children'].shape[0] - len(c),
                                       dtype=numpy.int16)))
        for c in children
    ]
    new_synsets['children'] = padded_children
    return new_synsets
