import os
import tarfile
import tempfile
import shutil
from collections import namedtuple

import h5py
import numpy
from scipy.io import loadmat
from six.moves import range
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar


FORMAT_1_FILES = ['{}.tar.gz'.format(s) for s in ['train', 'test', 'extra']]
FORMAT_1_TRAIN_FILE, FORMAT_1_TEST_FILE, FORMAT_1_EXTRA_FILE = FORMAT_1_FILES
FORMAT_2_FILES = ['{}_32x32.mat'.format(s) for s in ['train', 'test', 'extra']]
FORMAT_2_TRAIN_FILE, FORMAT_2_TEST_FILE, FORMAT_2_EXTRA_FILE = FORMAT_2_FILES


@check_exists(required_files=FORMAT_1_FILES)
def convert_svhn_format_1(directory, output_file):
    """Converts the SVHN dataset (format 1) to HDF5.

    This method assumes the existence of the files
    `{train,test,extra}.tar.gz`, which are accessible through the
    official website [SVHN].

    .. [SVHN] http://ufldl.stanford.edu/housenumbers/

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    h5file = h5py.File(output_file, mode='w')

    BoundingBoxes = namedtuple(
        'BoundingBoxes', ['heights', 'widths', 'lefts', 'tops'])

    TMPDIR = tempfile.mkdtemp()
    TRAIN_STRUCT = os.path.join(TMPDIR, 'train', 'digitStruct.mat')
    TEST_STRUCT = os.path.join(TMPDIR, 'test', 'digitStruct.mat')
    EXTRA_STRUCT = os.path.join(TMPDIR, 'extra', 'digitStruct.mat')

    def get_num_examples(tar_path):
        with tarfile.open(tar_path, 'r:gz') as f:
            num_examples = sum(1 for m in f.getmembers() if 'png' in m.name)
        return num_examples

    def extract_digit_struct(split, num_examples, struct_path, bar=None):
        boxes_and_labels = []
        with h5py.File(struct_path, 'r') as f:
            for i in range(num_examples):
                boxes_and_labels.append(get_bounding_boxes_and_labels(f, i))
                if bar:
                    bar.update(i)
        return boxes_and_labels

    def make_vlen_dataset(h5file, name, dtype, num_examples, shape_labels,
                          shapes_dtype='uint16'):
        dataset = h5file.create_dataset(
            name, (num_examples,),
            dtype=h5py.special_dtype(vlen=numpy.dtype(dtype)))
        dataset.dims[0].label = 'batch'
        dataset_shapes = h5file.create_dataset(
            '{}_shapes'.format(name), (num_examples, len(shape_labels)),
            dtype=shapes_dtype)
        dataset.dims.create_scale(dataset_shapes, 'shapes')
        dataset.dims[0].attach_scale(dataset_shapes)
        dataset_shape_labels = h5file.create_dataset(
            '{}_shape_labels'.format(name), (len(shape_labels),),
            dtype='S{}'.format(
                numpy.max([len(label) for label in shape_labels])))
        dataset_shape_labels[...] = [
            label.encode('utf8') for label in shape_labels]
        dataset.dims.create_scale(dataset_shape_labels, 'shape_labels')
        dataset.dims[0].attach_scale(dataset_shape_labels)

    def get_bounding_boxes_and_labels(h5file, image_number):
        bbox_group = h5file['digitStruct']['bbox'][image_number, 0]
        bbox_height_refs = h5file['digitStruct'][bbox_group]['height']
        bbox_width_refs = h5file['digitStruct'][bbox_group]['width']
        bbox_left_refs = h5file['digitStruct'][bbox_group]['left']
        bbox_top_refs = h5file['digitStruct'][bbox_group]['top']
        bbox_label_refs = h5file['digitStruct'][bbox_group]['label']

        num_boxes = len(bbox_height_refs)
        if num_boxes > 1:
            bounding_boxes = BoundingBoxes(
                heights=[int(h5file['digitStruct'][ref][0, 0])
                         for ref in bbox_height_refs[:, 0]],
                widths=[int(h5file['digitStruct'][ref][0, 0])
                        for ref in bbox_width_refs[:, 0]],
                lefts=[int(h5file['digitStruct'][ref][0, 0])
                       for ref in bbox_left_refs[:, 0]],
                tops=[int(h5file['digitStruct'][ref][0, 0])
                      for ref in bbox_top_refs[:, 0]])
            labels = [int(h5file['digitStruct'][ref][0, 0])
                      for ref in bbox_label_refs[:, 0]]
        else:
            bounding_boxes = BoundingBoxes(
                heights=[int(bbox_height_refs[0, 0])],
                widths=[int(bbox_width_refs[0, 0])],
                lefts=[int(bbox_left_refs[0, 0])],
                tops=[int(bbox_top_refs[0, 0])])
            labels = [int(bbox_label_refs[0, 0])]

        return bounding_boxes, labels

    def fill_split(h5file, split, num_examples, offset, base_path,
                   boxes_and_labels, bar=None):
        for image_number in range(num_examples):
            image_path = os.path.join(
                base_path, split, '{}.png'.format(image_number + 1))
            image = numpy.asarray(Image.open(image_path)).transpose(2, 0, 1)
            bounding_boxes, labels = boxes_and_labels[image_number]
            num_boxes = len(labels)
            i = image_number + offset

            h5file['features'][i] = image.flatten()
            h5file['features'].dims[0]['shapes'][i] = image.shape
            h5file['targets'][i] = labels
            h5file['targets'].dims[0]['shapes'][i] = len(labels)
            h5file['bbox_heights'][i] = bounding_boxes.heights
            h5file['bbox_heights'].dims[0]['shapes'][i] = num_boxes
            h5file['bbox_widths'][i] = bounding_boxes.widths
            h5file['bbox_widths'].dims[0]['shapes'][i] = num_boxes
            h5file['bbox_lefts'][i] = bounding_boxes.lefts
            h5file['bbox_lefts'].dims[0]['shapes'][i] = num_boxes
            h5file['bbox_tops'][i] = bounding_boxes.tops
            h5file['bbox_tops'].dims[0]['shapes'][i] = num_boxes

            if image_number % 1000 == 0:
                h5file.flush()
            if bar:
                bar.update(i)

    try:
        num_train = get_num_examples(FORMAT_1_TRAIN_FILE)
        num_test = get_num_examples(FORMAT_1_TEST_FILE)
        num_extra = get_num_examples(FORMAT_1_EXTRA_FILE)
        num_examples = num_train + num_test + num_extra

        print('Extracting train file...')
        with tarfile.open(FORMAT_1_TRAIN_FILE, 'r:gz') as f:
            f.extractall(path=TMPDIR)
        print('Extracting test file...')
        with tarfile.open(FORMAT_1_TEST_FILE, 'r:gz') as f:
            f.extractall(path=TMPDIR)
        print('Extracting extra file...')
        with tarfile.open(FORMAT_1_EXTRA_FILE, 'r:gz') as f:
            f.extractall(path=TMPDIR)

        with progress_bar('train digitStruct', num_train) as bar:
            train_boxes_and_labels = extract_digit_struct(
                split='train', num_examples=num_train,
                struct_path=TRAIN_STRUCT, bar=bar)
        with progress_bar('test digitStruct', num_test) as bar:
            test_boxes_and_labels = extract_digit_struct(
                split='test', num_examples=num_test,
                struct_path=TEST_STRUCT, bar=bar)
        with progress_bar('extra digitStruct', num_extra) as bar:
            extra_boxes_and_labels = extract_digit_struct(
                split='extra', num_examples=num_extra,
                struct_path=EXTRA_STRUCT, bar=bar)

        print('Setting up HDF5 file...')
        make_vlen_dataset(h5file=h5file, name='features', dtype='uint8',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('channel', 'height', 'width'))
        make_vlen_dataset(h5file=h5file, name='targets', dtype='uint8',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('bounding_box',))
        make_vlen_dataset(h5file=h5file, name='bbox_heights', dtype='uint16',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('bounding_box',))
        make_vlen_dataset(h5file=h5file, name='bbox_widths', dtype='uint16',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('bounding_box',))
        make_vlen_dataset(h5file=h5file, name='bbox_lefts', dtype='uint16',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('bounding_box',))
        make_vlen_dataset(h5file=h5file, name='bbox_tops', dtype='uint16',
                          num_examples=num_examples, shapes_dtype='uint16',
                          shape_labels=('bounding_box',))

        with progress_bar('SVHN format 1', num_examples) as bar:
            fill_split(
                split='train', h5file=h5file, num_examples=num_train, offset=0,
                base_path=TMPDIR, boxes_and_labels=train_boxes_and_labels,
                bar=bar)
            fill_split(
                split='test', h5file=h5file, num_examples=num_test,
                offset=num_train, base_path=TMPDIR,
                boxes_and_labels=test_boxes_and_labels, bar=bar)
            fill_split(
                split='extra', h5file=h5file, num_examples=num_extra,
                offset=num_train + num_test, base_path=TMPDIR,
                boxes_and_labels=extra_boxes_and_labels, bar=bar)
    finally:
        h5file.flush()
        h5file.close()
        if os.path.isdir(TMPDIR):
            shutil.rmtree(TMPDIR)


@check_exists(required_files=FORMAT_2_FILES)
def convert_svhn_format_2(directory, output_file):
    """Converts the SVHN dataset (format 2) to HDF5.

    This method assumes the existence of the files
    `{train,test,extra}_32x32.mat`, which are accessible through the
    official website [SVHN].

    .. [SVHN] http://ufldl.stanford.edu/housenumbers/

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    h5file = h5py.File(output_file, mode='w')

    train_set = loadmat(os.path.join(directory, FORMAT_2_TRAIN_FILE))
    train_features = train_set['X'].transpose(3, 2, 0, 1)
    train_targets = train_set['y']

    test_set = loadmat(os.path.join(directory, FORMAT_2_TEST_FILE))
    test_features = test_set['X'].transpose(3, 2, 0, 1)
    test_targets = test_set['y']

    extra_set = loadmat(os.path.join(directory, FORMAT_2_EXTRA_FILE))
    extra_features = extra_set['X'].transpose(3, 2, 0, 1)
    extra_targets = extra_set['y']

    data = (('train', 'features', train_features),
            ('test', 'features', test_features),
            ('extra', 'features', extra_features),
            ('train', 'targets', train_targets),
            ('test', 'targets', test_targets),
            ('extra', 'targets', extra_targets))
    fill_hdf5_file(h5file, data)
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label
    for i, label in enumerate(('batch', 'index')):
        h5file['targets'].dims[i].label = label

    h5file.flush()
    h5file.close()


def convert_svhn(which_format, directory, output_file):
    """Converts the SVHN dataset to HDF5.

    Converts the SVHN dataset [SVHN] to an HDF5 dataset compatible
    with :class:`fuel.datasets.SVHN`. The converted dataset is
    saved as 'svhn_format_1.hdf5' or 'svhn_format_2.hdf5', depending
    on the `which_format` argument.

    .. [SVHN] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco,
       Bo Wu, Andrew Y. Ng. *Reading Digits in Natural Images with
       Unsupervised Feature Learning*, NIPS Workshop on Deep Learning
       and Unsupervised Feature Learning, 2011.

    Parameters
    ----------
    which_format : int
        Either 1 or 2. Determines which format (format 1: full numbers
        or format 2: cropped digits) to convert.
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    if which_format not in (1, 2):
        raise ValueError("SVHN format needs to be either 1 or 2.")
    output_file = output_file.format(which_format)
    if which_format == 1:
        convert_svhn_format_1(directory, output_file)
    else:
        convert_svhn_format_2(directory, output_file)


def fill_subparser(subparser):
    """Sets up a subparser to convert the SVHN dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `svhn` command.

    """
    subparser.add_argument(
        "which_format", help="which dataset format", type=int, choices=(1, 2))
    subparser.set_defaults(
        func=convert_svhn,
        output_file=os.path.join(os.getcwd(), 'svhn_format_{}.hdf5'))
