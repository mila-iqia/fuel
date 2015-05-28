import os
import tarfile
import tempfile
import shutil
from collections import namedtuple, OrderedDict

import h5py
import numpy
from scipy.io import loadmat
from six.moves import range
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar
from fuel.datasets import H5PYDataset


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
    try:
        h5file = h5py.File(output_file, mode='w')
        TMPDIR = tempfile.mkdtemp()

        BoundingBoxes = namedtuple(
            'BoundingBoxes', ['heights', 'widths', 'lefts', 'tops'])

        splits = ('train', 'test', 'extra')
        file_paths = dict(zip(splits, FORMAT_1_FILES))
        split_structs = dict(
            [(split, os.path.join(TMPDIR, split, 'digitStruct.mat'))
             for split in splits])

        sources = ('features', 'targets', 'bbox_heights', 'bbox_widths',
                   'bbox_lefts', 'bbox_tops')
        source_dtypes = dict([(source, 'uint8') for source in sources[:2]] +
                             [(source, 'uint16') for source in sources[2:]])
        source_shape_labels = dict(
            [('features', ('channel', 'height', 'width')),
             ('targets', ('index',))] +
            [(source, ('bounding_box',)) for source in sources[2:]])

        def extract_tar(split):
            with tarfile.open(file_paths[split], 'r:gz') as f:
                members = f.getmembers()
                num_examples = sum(1 for m in members if '.png' in m.name)
                progress_bar_context = progress_bar(
                    name='{} file'.format(split), maxval=len(members),
                    prefix='Extracting')
                with progress_bar_context as bar:
                    for i, member in enumerate(members):
                        f.extract(member, path=TMPDIR)
                        bar.update(i)
            return num_examples

        examples_per_split = {}
        for split in splits:
            examples_per_split[split] = extract_tar(split)

        num_examples = sum(examples_per_split.values())
        split_intervals = {}
        split_intervals['train'] = (0, examples_per_split['train'])
        split_intervals['test'] = (
            examples_per_split['train'],
            examples_per_split['train'] + examples_per_split['test'])
        split_intervals['extra'] = (
            examples_per_split['train'] + examples_per_split['test'],
            num_examples)
        split_dict = OrderedDict([
            (split, OrderedDict([(s, split_intervals[split])
                                 for s in sources]))
            for split in splits])
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        def make_vlen_dataset(source):
            dtype = h5py.special_dtype(vlen=numpy.dtype(source_dtypes[source]))
            dataset = h5file.create_dataset(
                source, (num_examples,), dtype=dtype)
            dataset.dims[0].label = 'batch'
            shape_labels = source_shape_labels[source]
            dataset_shapes = h5file.create_dataset(
                '{}_shapes'.format(source), (num_examples, len(shape_labels)),
                dtype='uint16')
            dataset.dims.create_scale(dataset_shapes, 'shapes')
            dataset.dims[0].attach_scale(dataset_shapes)
            dataset_shape_labels = h5file.create_dataset(
                '{}_shape_labels'.format(source), (len(shape_labels),),
                dtype='S{}'.format(
                    numpy.max([len(label) for label in shape_labels])))
            dataset_shape_labels[...] = [
                label.encode('utf8') for label in shape_labels]
            dataset.dims.create_scale(dataset_shape_labels, 'shape_labels')
            dataset.dims[0].attach_scale(dataset_shape_labels)

        for source in sources:
            make_vlen_dataset(source)

        def get_boxes_and_labels(split):
            boxes_and_labels = []
            num_ex = examples_per_split[split]
            progress_bar_context = progress_bar(
                '{} digitStruct'.format(split), num_ex)
            path = split_structs[split]
            with h5py.File(path, 'r') as f, progress_bar_context as bar:
                for image_number in range(num_ex):
                    bbox_group = f['digitStruct']['bbox'][image_number, 0]
                    bbox_height_refs = f['digitStruct'][bbox_group]['height']
                    bbox_width_refs = f['digitStruct'][bbox_group]['width']
                    bbox_left_refs = f['digitStruct'][bbox_group]['left']
                    bbox_top_refs = f['digitStruct'][bbox_group]['top']
                    bbox_label_refs = f['digitStruct'][bbox_group]['label']

                    num_boxes = len(bbox_height_refs)
                    if num_boxes > 1:
                        bounding_boxes = BoundingBoxes(
                            heights=[int(f['digitStruct'][ref][0, 0])
                                     for ref in bbox_height_refs[:, 0]],
                            widths=[int(f['digitStruct'][ref][0, 0])
                                    for ref in bbox_width_refs[:, 0]],
                            lefts=[int(f['digitStruct'][ref][0, 0])
                                   for ref in bbox_left_refs[:, 0]],
                            tops=[int(f['digitStruct'][ref][0, 0])
                                  for ref in bbox_top_refs[:, 0]])
                        labels = [int(f['digitStruct'][ref][0, 0])
                                  for ref in bbox_label_refs[:, 0]]
                    else:
                        bounding_boxes = BoundingBoxes(
                            heights=[int(bbox_height_refs[0, 0])],
                            widths=[int(bbox_width_refs[0, 0])],
                            lefts=[int(bbox_left_refs[0, 0])],
                            tops=[int(bbox_top_refs[0, 0])])
                        labels = [int(bbox_label_refs[0, 0])]

                    boxes_and_labels.append((bounding_boxes, labels))
                    if bar:
                        bar.update(image_number)
            return boxes_and_labels

        split_boxes_and_labels = {}
        for split in splits:
            split_boxes_and_labels[split] = get_boxes_and_labels(split)

        def fill_split(split, bar=None):
            for image_number in range(examples_per_split[split]):
                image_path = os.path.join(
                    TMPDIR, split, '{}.png'.format(image_number + 1))
                image = numpy.asarray(
                    Image.open(image_path)).transpose(2, 0, 1)
                rval = split_boxes_and_labels[split][image_number]
                bounding_boxes, labels = rval
                num_boxes = len(labels)
                i = image_number + split_intervals[split][0]

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

        with progress_bar('SVHN format 1', num_examples) as bar:
            for split in splits:
                fill_split(split, bar=bar)
    finally:
        if os.path.isdir(TMPDIR):
            shutil.rmtree(TMPDIR)
        h5file.flush()
        h5file.close()


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
