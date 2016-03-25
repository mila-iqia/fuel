import os
import zipfile

import h5py
import numpy
from PIL import Image

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets.hdf5 import H5PYDataset

TRAIN = 'dogs_vs_cats.train.zip'
TEST = 'dogs_vs_cats.test1.zip'


@check_exists(required_files=[TRAIN, TEST])
def convert_dogs_vs_cats(directory, output_directory,
                         output_filename='dogs_vs_cats.hdf5'):
    """Converts the Dogs vs. Cats dataset to HDF5.

    Converts the Dogs vs. Cats dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.dogs_vs_cats`. The converted dataset is saved as
    'dogs_vs_cats.hdf5'.

    It assumes the existence of the following files:

    * `dogs_vs_cats.train.zip`
    * `dogs_vs_cats.test1.zip`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'dogs_vs_cats.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    # Prepare output file
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    hdf_features = h5file.create_dataset('image_features', (37500,),
                                         dtype=dtype)
    hdf_shapes = h5file.create_dataset('image_features_shapes', (37500, 3),
                                       dtype='int32')
    hdf_labels = h5file.create_dataset('targets', (25000, 1), dtype='uint8')

    # Attach shape annotations and scales
    hdf_features.dims.create_scale(hdf_shapes, 'shapes')
    hdf_features.dims[0].attach_scale(hdf_shapes)

    hdf_shapes_labels = h5file.create_dataset('image_features_shapes_labels',
                                              (3,), dtype='S7')
    hdf_shapes_labels[...] = ['channel'.encode('utf8'),
                              'height'.encode('utf8'),
                              'width'.encode('utf8')]
    hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
    hdf_features.dims[0].attach_scale(hdf_shapes_labels)

    # Add axis annotations
    hdf_features.dims[0].label = 'batch'
    hdf_labels.dims[0].label = 'batch'
    hdf_labels.dims[1].label = 'index'

    # Convert
    i = 0
    for split, split_size in zip([TRAIN, TEST], [25000, 12500]):
        # Open the ZIP file
        filename = os.path.join(directory, split)
        zip_file = zipfile.ZipFile(filename, 'r')
        image_names = zip_file.namelist()[1:]  # Discard the directory name

        # Shuffle the examples
        if split == TRAIN:
            rng = numpy.random.RandomState(123522)
            rng.shuffle(image_names)
        else:
            image_names.sort(key=lambda fn: int(os.path.splitext(fn[6:])[0]))

        # Convert from JPEG to NumPy arrays
        with progress_bar(filename, split_size) as bar:
            for image_name in image_names:
                # Save image
                image = numpy.array(Image.open(zip_file.open(image_name)))
                image = image.transpose(2, 0, 1)
                hdf_features[i] = image.flatten()
                hdf_shapes[i] = image.shape

                # Cats are 0, Dogs are 1
                if split == TRAIN:
                    hdf_labels[i] = 0 if 'cat' in image_name else 1

                # Update progress
                i += 1
                bar.update(i if split == TRAIN else i - 25000)

    # Add the labels
    split_dict = {}
    sources = ['image_features', 'targets']
    split_dict['train'] = dict(zip(sources, [(0, 25000)] * 2))
    split_dict['test'] = {sources[0]: (25000, 37500)}
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the dogs_vs_cats dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `dogs_vs_cats` command.

    """
    return convert_dogs_vs_cats
