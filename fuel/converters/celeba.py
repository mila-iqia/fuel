import os
import zipfile

import h5py
import numpy
from six.moves import range
from PIL import Image

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets import H5PYDataset

IMAGE_FILE = 'img_align_celeba.zip'
ATTRIBUTES_FILE = 'list_attr_celeba.txt'
DATASET_FILES = [IMAGE_FILE, ATTRIBUTES_FILE]
NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637
OUTPUT_FILENAME = 'celeba_aligned_cropped.hdf5'


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
        'targets', (NUM_EXAMPLES, 40), dtype='uint8')
    targets_dataset.dims[0].label = 'batch'
    targets_dataset.dims[1].label = 'target'
    targets_dataset[...] = (
        numpy.loadtxt(os.path.join(directory, ATTRIBUTES_FILE), dtype='int32',
                      skiprows=2, usecols=tuple(range(1, 41))) +
        1) / 2

    features_dataset = h5file.create_dataset(
        'features', (NUM_EXAMPLES, 3) + image_shape, dtype='uint8')
    features_dataset.dims[0].label = 'batch'
    features_dataset.dims[1].label = 'channel'
    features_dataset.dims[2].label = 'height'
    features_dataset.dims[3].label = 'width'

    return h5file


@check_exists(required_files=DATASET_FILES)
def convert_celeba_aligned_cropped(directory, output_directory,
                                   output_filename=OUTPUT_FILENAME):
    """Converts the aligned and cropped CelebA dataset to HDF5.

    Converts the CelebA dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CelebA`. The converted dataset is saved as
    'celeba_aligned_cropped.hdf5'.

    It assumes the existence of the following files:

    * `img_align_celeba.zip`
    * `list_attr_celeba.txt`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to
        'celeba_aligned_cropped.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted
        dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = _initialize_conversion(directory, output_path, (218, 178))

    features_dataset = h5file['features']
    image_file_path = os.path.join(directory, IMAGE_FILE)
    with zipfile.ZipFile(image_file_path, 'r') as image_file:
        with progress_bar('images', NUM_EXAMPLES) as bar:
            for i in range(NUM_EXAMPLES):
                image_name = 'img_align_celeba/{:06d}.jpg'.format(i + 1)
                features_dataset[i] = numpy.asarray(
                    Image.open(
                        image_file.open(image_name, 'r'))).transpose(2, 0, 1)
                bar.update(i + 1)

    h5file.flush()
    h5file.close()

    return (output_path,)


@check_exists(required_files=DATASET_FILES)
def convert_celeba_64(directory, output_directory,
                      output_filename='celeba_64.hdf5'):
    """Converts the 64x64 version of the CelebA dataset to HDF5.

    This converter takes the aligned and cropped version of the
    CelebA dataset as input and produces a version that's been resized
    to 78x64 pixels and then center cropped to 64x64 pixels.

    Converts the CelebA dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CelebA`. The converted dataset is saved as
    'celeba_64.hdf5'.

    It assumes the existence of the following files:

    * `img_align_celeba.zip`
    * `list_attr_celeba.txt`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'celeba_64.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = _initialize_conversion(directory, output_path, (64, 64))

    features_dataset = h5file['features']
    image_file_path = os.path.join(directory, IMAGE_FILE)
    with zipfile.ZipFile(image_file_path, 'r') as image_file:
        with progress_bar('images', NUM_EXAMPLES) as bar:
            for i in range(NUM_EXAMPLES):
                image_name = 'img_align_celeba/{:06d}.jpg'.format(i + 1)
                image = Image.open(
                    image_file.open(image_name, 'r')).resize(
                        (64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
                features_dataset[i] = numpy.asarray(image).transpose(2, 0, 1)
                bar.update(i + 1)

    h5file.flush()
    h5file.close()

    return (output_path,)


def convert_celeba(which_format, directory, output_directory,
                   output_filename=None):
    """Converts the CelebA dataset to HDF5.

    Converts the CelebA dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CelebA`. The converted dataset is
    saved as 'celeba_aligned_cropped.hdf5' or 'celeba_64.hdf5',
    depending on the `which_format` argument.

    Parameters
    ----------
    which_format : str
        Either 'aligned_cropped' or '64'. Determines which format
        to convert to.
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to
        'celeba_aligned_cropped.hdf5' or 'celeba_64.hdf5',
        depending on `which_format`.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    if which_format not in ('aligned_cropped', '64'):
        raise ValueError("CelebA format needs to be either "
                         "'aligned_cropped' or '64'.")
    if not output_filename:
        output_filename = 'celeba_{}.hdf5'.format(which_format)
    if which_format == 'aligned_cropped':
        return convert_celeba_aligned_cropped(
            directory, output_directory, output_filename)
    else:
        return convert_celeba_64(
            directory, output_directory, output_filename)


def fill_subparser(subparser):
    """Sets up a subparser to convert the CelebA dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `celeba` command.

    """
    subparser.add_argument(
        "which_format", help="which dataset format", type=str,
        choices=('aligned_cropped', '64'))
    return convert_celeba
