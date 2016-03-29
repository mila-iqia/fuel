import os
import h5py

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file, MissingInputFiles


def convert_silhouettes(size, directory, output_directory,
                        output_filename=None):
    """ Convert the CalTech 101 Silhouettes Datasets.

    Parameters
    ----------
    size : {16, 28}
        Convert either the 16x16 or 28x28 sized version of the dataset.
    directory : str
        Directory in which the required input files reside.
    output_filename : str
        Where to save the converted dataset.

    """
    if size not in (16, 28):
        raise ValueError('size must be 16 or 28')

    if output_filename is None:
        output_filename = 'caltech101_silhouettes{}.hdf5'.format(size)
    output_file = os.path.join(output_directory, output_filename)

    input_file = 'caltech101_silhouettes_{}_split1.mat'.format(size)
    input_file = os.path.join(directory, input_file)

    if not os.path.isfile(input_file):
        raise MissingInputFiles('Required files missing', [input_file])

    with h5py.File(output_file, mode="w") as h5file:
        mat = loadmat(input_file)

        train_features = mat['train_data'].reshape([-1, 1, size, size])
        train_targets = mat['train_labels']
        valid_features = mat['val_data'].reshape([-1, 1, size, size])
        valid_targets = mat['val_labels']
        test_features = mat['test_data'].reshape([-1, 1, size, size])
        test_targets = mat['test_labels']

        data = (
            ('train', 'features', train_features),
            ('train', 'targets', train_targets),
            ('valid', 'features', valid_features),
            ('valid', 'targets', valid_targets),
            ('test', 'features', test_features),
            ('test', 'targets', test_targets),
        )
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['targets'].dims[i].label = label
    return (output_file,)


def fill_subparser(subparser):
    """Sets up a subparser to convert CalTech101 Silhouettes Database files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `caltech101_silhouettes` command.

    """
    subparser.add_argument(
        "size", type=int, choices=(16, 28),
        help="height/width of the datapoints")
    return convert_silhouettes
