import os
import h5py

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file, check_exists

ALL_FILES16 = ['caltech101_silhouettes_16_split1.mat'] #, 'caltech101_silhouettes_16_split1.mat']
ALL_FILES28 = ['caltech101_silhouettes_28_split1.mat'] #, 'caltech101_silhouettes_28_split1.mat']


# TODO: Handel 16x16 dataset

@check_exists(required_files=ALL_FILES28)
def convert_caltech101_silhouettes28(directory, output_file):
    """ Convert the CalTech 101 Silhouettes Datasets.

    ToDo

    Parameters
    ----------
    directory : str
        Directory in which the required input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    with h5py.File(output_file, mode="w") as h5file:
        tfd = loadmat(os.path.join(directory, 'caltech101_silhouettes_28_split1.mat'))

        train_X = tfd['train_data'].reshape([-1., 1, 28, 28])
        valid_X = tfd['val_data'].reshape([-1., 1, 28, 28])
        test_X  = tfd['test_data'].reshape([-1., 1, 28, 28])
        train_Y = tfd['train_labels']
        valid_Y = tfd['val_labels']
        test_Y  = tfd['test_labels']

        data = (
            ('train', 'features', train_X),
            ('train', 'targets' , train_Y),
            ('valid', 'features', valid_X),
            ('valid', 'targets' , valid_Y),
            ('test',  'features', test_X),
            ('test',  'targets' , test_Y),
        )
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['targets'].dims[i].label = label


def fill_subparser(subparser):
    """Sets up a subparser to convert Toronto Face Database files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `caltech101_silhouettes` command.

    """
    subparser.set_defaults(func=convert_caltech101_silhouettes28)
