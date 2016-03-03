import os
import tarfile
import tempfile
import shutil
from collections import namedtuple, OrderedDict

import h5py
import numpy
from scipy.io import loadmat
from six import iteritems
from six.moves import range, zip
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar
from fuel.datasets import H5PYDataset


FORMAT_1_FILES = ['{}.tar.gz'.format(s) for s in ['train', 'test', 'extra']]
FORMAT_1_TRAIN_FILE, FORMAT_1_TEST_FILE, FORMAT_1_EXTRA_FILE = FORMAT_1_FILES
FORMAT_2_FILES = ['{}_32x32.mat'.format(s) for s in ['train', 'test', 'extra']]
FORMAT_2_TRAIN_FILE, FORMAT_2_TEST_FILE, FORMAT_2_EXTRA_FILE = FORMAT_2_FILES


@check_exists(required_files=FORMAT_1_FILES)
def convert_svhn_format_1(directory, output_directory,
                          output_filename='svhn_format_1.hdf5'):
    """Converts the SVHN dataset (format 1) to HDF5.

    This method assumes the existence of the files
    `{train,test,extra}.tar.gz`, which are accessible through the
    official website [SVHNSITE].

    .. [SVHNSITE] http://ufldl.stanford.edu/housenumbers/

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'svhn_format_1.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    try:
        output_path = os.path.join(output_directory, output_filename)
        h5file = h5py.File(output_path, mode='w')
        TMPDIR = tempfile.mkdtemp()

        # Every image has three channels (RGB) and variable height and width.
        # It features a variable number of bounding boxes that identify the
        # location and label of digits. The bounding box location is specified
        # using the x and y coordinates of its top left corner along with its
        # width and height.
        BoundingBoxes = namedtuple(
            'BoundingBoxes', ['labels', 'heights', 'widths', 'lefts', 'tops'])
        sources = ('features',) + tuple('bbox_{}'.format(field)
                                        for field in BoundingBoxes._fields)
        source_dtypes = dict([(source, 'uint8') for source in sources[:2]] +
                             [(source, 'uint16') for source in sources[2:]])
        source_axis_labels = {
            'features': ('channel', 'height', 'width'),
            'bbox_labels': ('bounding_box', 'index'),
            'bbox_heights': ('bounding_box', 'height'),
            'bbox_widths': ('bounding_box', 'width'),
            'bbox_lefts': ('bounding_box', 'x'),
            'bbox_tops': ('bounding_box', 'y')}

        # The dataset is split into three sets: the training set, the test set
        # and an extra set of examples that are somewhat less difficult but
        # can be used as extra training data. These sets are stored separately
        # as 'train.tar.gz', 'test.tar.gz' and 'extra.tar.gz'. Each file
        # contains a directory named after the split it stores. The examples
        # are stored in that directory as PNG images. The directory also
        # contains a 'digitStruct.mat' file with all the bounding box and
        # label information.
        splits = ('train', 'test', 'extra')
        file_paths = dict(zip(splits, FORMAT_1_FILES))
        for split, path in file_paths.items():
            file_paths[split] = os.path.join(directory, path)
        digit_struct_paths = dict(
            [(split, os.path.join(TMPDIR, split, 'digitStruct.mat'))
             for split in splits])

        # We first extract the data files in a temporary directory. While doing
        # that, we also count the number of examples for each split. Files are
        # extracted individually, which allows to display a progress bar. Since
        # the splits will be concatenated in the HDF5 file, we also compute the
        # start and stop intervals of each split within the concatenated array.
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

        examples_per_split = OrderedDict(
            [(split, extract_tar(split)) for split in splits])
        cumulative_num_examples = numpy.cumsum(
            [0] + list(examples_per_split.values()))
        num_examples = cumulative_num_examples[-1]
        intervals = zip(cumulative_num_examples[:-1],
                        cumulative_num_examples[1:])
        split_intervals = dict(zip(splits, intervals))

        # The start and stop indices are used to create a split dict that will
        # be parsed into the split array required by the H5PYDataset interface.
        # The split dict is organized as follows:
        #
        #     dict(split -> dict(source -> (start, stop)))
        #
        split_dict = OrderedDict([
            (split, OrderedDict([(s, split_intervals[split])
                                 for s in sources]))
            for split in splits])
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        # We then prepare the HDF5 dataset. This involves creating datasets to
        # store data sources and datasets to store auxiliary information
        # (namely the shapes for variable-length axes, and labels to indicate
        # what these variable-length axes represent).
        def make_vlen_dataset(source):
            # Create a variable-length 1D dataset
            dtype = h5py.special_dtype(vlen=numpy.dtype(source_dtypes[source]))
            dataset = h5file.create_dataset(
                source, (num_examples,), dtype=dtype)
            # Create a dataset to store variable-length shapes.
            axis_labels = source_axis_labels[source]
            dataset_shapes = h5file.create_dataset(
                '{}_shapes'.format(source), (num_examples, len(axis_labels)),
                dtype='uint16')
            # Create a dataset to store labels for variable-length axes.
            dataset_vlen_axis_labels = h5file.create_dataset(
                '{}_vlen_axis_labels'.format(source), (len(axis_labels),),
                dtype='S{}'.format(
                    numpy.max([len(label) for label in axis_labels])))
            # Fill variable-length axis labels
            dataset_vlen_axis_labels[...] = [
                label.encode('utf8') for label in axis_labels]
            # Attach auxiliary datasets as dimension scales of the
            # variable-length 1D dataset. This is in accordance with the
            # H5PYDataset interface.
            dataset.dims.create_scale(dataset_shapes, 'shapes')
            dataset.dims[0].attach_scale(dataset_shapes)
            dataset.dims.create_scale(dataset_vlen_axis_labels, 'shape_labels')
            dataset.dims[0].attach_scale(dataset_vlen_axis_labels)
            # Tag fixed-length axis with its label
            dataset.dims[0].label = 'batch'

        for source in sources:
            make_vlen_dataset(source)

        # The "fun" part begins: we extract the bounding box and label
        # information contained in 'digitStruct.mat'. This is a version 7.3
        # Matlab file, which uses HDF5 under the hood, albeit with a very
        # convoluted layout.
        def get_boxes(split):
            boxes = []
            with h5py.File(digit_struct_paths[split], 'r') as f:
                bar_name = '{} digitStruct'.format(split)
                bar_maxval = examples_per_split[split]
                with progress_bar(bar_name, bar_maxval) as bar:
                    for image_number in range(examples_per_split[split]):
                        # The 'digitStruct' group is the main group of the HDF5
                        # file. It contains two datasets: 'bbox' and 'name'.
                        # The 'name' dataset isn't of interest to us, as it
                        # stores file names and there's already a one-to-one
                        # mapping between row numbers and image names (e.g.
                        # row 0 corresponds to '1.png', row 1 corresponds to
                        # '2.png', and so on).
                        main_group = f['digitStruct']
                        # The 'bbox' dataset contains the bounding box and
                        # label information we're after. It has as many rows
                        # as there are images, and one column. Elements of the
                        # 'bbox' dataset are object references that point to
                        # (yet another) group that contains the information
                        # for the corresponding image.
                        image_reference = main_group['bbox'][image_number, 0]

                        # There are five datasets contained in that group:
                        # 'label', 'height', 'width', 'left' and 'top'. Each of
                        # those datasets has as many rows as there are bounding
                        # boxes in the corresponding image, and one column.
                        def get_dataset(name):
                            return main_group[image_reference][name][:, 0]
                        names = ('label', 'height', 'width', 'left', 'top')
                        datasets = dict(
                            [(name, get_dataset(name)) for name in names])

                        # If there is only one bounding box, the information is
                        # stored directly in the datasets. If there are
                        # multiple bounding boxes, elements of those datasets
                        # are object references pointing to 1x1 datasets that
                        # store the information (fortunately, it's the last
                        # hop we need to make).
                        def get_elements(dataset):
                            if len(dataset) > 1:
                                return [int(main_group[reference][0, 0])
                                        for reference in dataset]
                            else:
                                return [int(dataset[0])]
                        # Names are pluralized in the BoundingBox named tuple.
                        kwargs = dict(
                            [(name + 's', get_elements(dataset))
                             for name, dataset in iteritems(datasets)])
                        boxes.append(BoundingBoxes(**kwargs))
                        if bar:
                            bar.update(image_number)
            return boxes

        split_boxes = dict([(split, get_boxes(split)) for split in splits])

        # The final step is to fill the HDF5 file.
        def fill_split(split, bar=None):
            for image_number in range(examples_per_split[split]):
                image_path = os.path.join(
                    TMPDIR, split, '{}.png'.format(image_number + 1))
                image = numpy.asarray(
                    Image.open(image_path)).transpose(2, 0, 1)
                bounding_boxes = split_boxes[split][image_number]
                num_boxes = len(bounding_boxes.labels)
                index = image_number + split_intervals[split][0]

                h5file['features'][index] = image.flatten()
                h5file['features'].dims[0]['shapes'][index] = image.shape
                for field in BoundingBoxes._fields:
                    name = 'bbox_{}'.format(field)
                    h5file[name][index] = numpy.maximum(0,
                                                        getattr(bounding_boxes,
                                                                field))
                    h5file[name].dims[0]['shapes'][index] = [num_boxes, 1]

                # Replace label '10' with '0'.
                labels = h5file['bbox_labels'][index]
                labels[labels == 10] = 0
                h5file['bbox_labels'][index] = labels

                if image_number % 1000 == 0:
                    h5file.flush()
                if bar:
                    bar.update(index)

        with progress_bar('SVHN format 1', num_examples) as bar:
            for split in splits:
                fill_split(split, bar=bar)
    finally:
        if os.path.isdir(TMPDIR):
            shutil.rmtree(TMPDIR)
        h5file.flush()
        h5file.close()

    return (output_path,)


@check_exists(required_files=FORMAT_2_FILES)
def convert_svhn_format_2(directory, output_directory,
                          output_filename='svhn_format_2.hdf5'):
    """Converts the SVHN dataset (format 2) to HDF5.

    This method assumes the existence of the files
    `{train,test,extra}_32x32.mat`, which are accessible through the
    official website [SVHNSITE].

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'svhn_format_2.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')

    train_set = loadmat(os.path.join(directory, FORMAT_2_TRAIN_FILE))
    train_features = train_set['X'].transpose(3, 2, 0, 1)
    train_targets = train_set['y']
    train_targets[train_targets == 10] = 0

    test_set = loadmat(os.path.join(directory, FORMAT_2_TEST_FILE))
    test_features = test_set['X'].transpose(3, 2, 0, 1)
    test_targets = test_set['y']
    test_targets[test_targets == 10] = 0

    extra_set = loadmat(os.path.join(directory, FORMAT_2_EXTRA_FILE))
    extra_features = extra_set['X'].transpose(3, 2, 0, 1)
    extra_targets = extra_set['y']
    extra_targets[extra_targets == 10] = 0

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

    return (output_path,)


def convert_svhn(which_format, directory, output_directory,
                 output_filename=None):
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
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'svhn_format_1.hdf5' or
        'svhn_format_2.hdf5', depending on `which_format`.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    if which_format not in (1, 2):
        raise ValueError("SVHN format needs to be either 1 or 2.")
    if not output_filename:
        output_filename = 'svhn_format_{}.hdf5'.format(which_format)
    if which_format == 1:
        return convert_svhn_format_1(
            directory, output_directory, output_filename)
    else:
        return convert_svhn_format_2(
            directory, output_directory, output_filename)


def fill_subparser(subparser):
    """Sets up a subparser to convert the SVHN dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `svhn` command.

    """
    subparser.add_argument(
        "which_format", help="which dataset format", type=int, choices=(1, 2))
    return convert_svhn
