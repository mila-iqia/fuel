import os

import h5py

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file, check_exists

ALL_FILES = ['TFD_48x48.mat']


@check_exists(required_files=ALL_FILES)
def convert_toronto_face_database(input_directory, output_fname):
    """Converts the Toronto Face Database to HDF5.

    ToDo

    Parameters
    ----------
    input_directory : str
        Directory in which the required input files reside.
    output_fname : str
        Where to save the converted dataset.

    """
    with h5py.File(output_fname, mode="w") as h5file:

        tfd = loadmat(os.path.join(input_directory, 'TFD_48x48.mat'))

        folds = tfd['folds']
        features = tfd['images'].reshape([-1, 1, 48, 48])
        expression_targets = tfd['labs_ex']
        identity_targets = tfd['labs_id']

        unlabeled_mask = folds[:, 0] == 0
        train1_mask = folds[:, 0] == 1
        valid1_mask = folds[:, 0] == 2
        test1_mask = folds[:, 0] == 3

        data = (
            ('unlabeled', 'features', features[unlabeled_mask]),
            ('train', 'features', features[train1_mask]),
            ('train', 'expression_targets', expression_targets[train1_mask]),
            ('train', 'identity_targets', identity_targets[train1_mask]),
            ('valid', 'features', features[valid1_mask]),
            ('valid', 'expression_targets', expression_targets[valid1_mask]),
            ('valid', 'identity_targets', identity_targets[valid1_mask]),
            ('test', 'features', features[test1_mask]),
            ('test', 'expression_targets', expression_targets[test1_mask]),
            ('test', 'identity_targets', identity_targets[test1_mask]),
        )
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['expression_targets'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['identity_targets'].dims[i].label = label


def fill_subparser(subparser):
    """Sets up a subparser to convert Toronto Face Database files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `toronto_face_database` command.

    """
    subparser.set_defaults(func=convert_toronto_face_database)
