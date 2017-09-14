import os

import h5py
import numpy
from six.moves import range
from PIL import Image

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets import H5PYDataset

IMAGE_FOLDER = 'Camvid'
ATTRIBUTES_FILE_TRAIN = 'train.txt'
ATTRIBUTES_FILE_VALID = 'val.txt'
ATTRIBUTES_FILE_TEST = 'test.txt'
DATASET_FILES = [ATTRIBUTES_FILE_TRAIN, ATTRIBUTES_FILE_VALID, ATTRIBUTES_FILE_TEST]
NUM_EXAMPLES = 701
TRAIN_STOP = 367
VALID_STOP = 468
OUTPUT_FILENAME = 'camvid.hdf5'


def _initialize_conversion(directory, output_path, image_shape):
    h5file = h5py.File(output_path, mode='w')
    split_dict = {
        'train': {
            'features': (0, TRAIN_STOP),
            'targets': (0, TRAIN_STOP)},
        'valid': {
            'features': (TRAIN_STOP, VALID_STOP),
            'targets': (TRAIN_STOP, VALID_STOP)},
        'test': {
            'features': (VALID_STOP, NUM_EXAMPLES),
            'targets': (VALID_STOP, NUM_EXAMPLES)}}
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    targets_dataset = h5file.create_dataset(
        'targets', (NUM_EXAMPLES,) + image_shape, dtype='uint8')
    targets_dataset.dims[0].label = 'batch'
    targets_dataset.dims[1].label = 'height'
    targets_dataset.dims[2].label = 'width'

    features_dataset = h5file.create_dataset(
        'features', (NUM_EXAMPLES, 3) + image_shape, dtype='uint8')
    features_dataset.dims[0].label = 'batch'
    features_dataset.dims[1].label = 'channel'
    features_dataset.dims[2].label = 'height'
    features_dataset.dims[3].label = 'width'

    return h5file

@check_exists(required_files=DATASET_FILES)
def convert_camvid(directory, output_directory,
                      output_filename='camvid.hdf5'):
    """Converts the camvid dataset to HDF5.

    Converts the camvid dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.camvid`. The converted dataset is
    saved as 'camvid.hdf5'.

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to
        'camvid_aligned_cropped.hdf5' or 'camvid_64.hdf5',
        depending on `which_format`.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = _initialize_conversion(directory, output_path, (360, 480))

    features_dataset = h5file['features']
    targets_dataset = h5file['targets']
    with progress_bar('images', NUM_EXAMPLES) as bar:
        for files in DATASET_FILES:
            open_file = open(files, 'r')
            for i, line in enumerate(open_file):
                image_name, target_name = line.split()
                image = Image.open(image_name[15:], 'r')
                target = Image.open(target_name[15:], 'r')
                features_dataset[i] = numpy.asarray(image).transpose(2, 0, 1)
                targets_dataset[i] = numpy.asarray(target)
                bar.update(i + 1)

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the Camvid dataset files.
    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `camvid` command.
    """
    return convert_camvid
