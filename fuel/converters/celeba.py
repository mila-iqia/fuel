import os
import zipfile

import h5py
import numpy
import six
from six.moves import range, cPickle
from PIL import Image

from fuel.converters.base import check_exists, progress_bar

IMAGE_FILE = 'img_align_celeba.zip'
ATTRIBUTES_FILE = 'list_attr_celeba.txt'
DATASET_FILES = [IMAGE_FILE, ATTRIBUTES_FILE]
NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637


@check_exists(required_files=DATASET_FILES)
def convert_celeba(directory, output_directory, output_filename='celeba.hdf5'):
    """Converts the CelebA dataset to HDF5.

    Converts the CelebA dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CelebA`. The converted dataset is saved as
    'celeba.hdf5'.

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
        Name of the saved dataset. Defaults to 'celeba.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')

    attributes_dataset = h5file.create_dataset(
        'attributes', (NUM_EXAMPLES, 40), dtype='uint8')
    attributes_dataset[...] = (
        numpy.loadtxt(os.path.join(directory, ATTRIBUTES_FILE), dtype='int32',
                      skiprows=2, usecols=tuple(range(1, 41)))
        + 1) / 2

    features_dataset = h5file.create_dataset(
        'features', (NUM_EXAMPLES, 3, 218, 178), dtype='uint8')
    image_file = zipfile.ZipFile(os.path.join(directory, IMAGE_FILE), 'r')
    with progress_bar('images', NUM_EXAMPLES) as bar:
        for i in range(NUM_EXAMPLES):
            image_name = 'img_align_celeba/{:06d}.jpg'.format(i + 1)
            features_dataset[i] = numpy.asarray(
                Image.open(image_file.open(image_name, 'r'))).transpose(2, 0, 1)
            bar.update(i + 1)

    split_dict = {
        'train': {
            'features': (0, TRAIN_STOP),
            'attributes': (0, TRAIN_STOP)},
        'valid': {
            'features': (TRAIN_STOP, VALID_STOP),
            'attributes': (TRAIN_STOP, VALID_STOP)},
        'test': {
            'features': (VALID_STOP, NUM_EXAMPLES),
            'attributes': (VALID_STOP, NUM_EXAMPLES)}}
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['attributes'].dims[0].label = 'batch'
    h5file['attributes'].dims[1].label = 'attribute'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the CelebA dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `celeba` command.

    """
    return convert_celeba
