import os
import h5py

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file, MissingInputFiles

TFD_INPUT_FILE = 'TFD_{}x{}.mat'


def convert_tfd(size, directory, output_file):
    """Converts the Toronto Face Database to HDF5.

    ToDo

    Parameters
    ----------
    size : int
        Either 48 or 96. Indicates whether the dataset containing 48x48 sized
        images should be converted, or the one containing 96x96 sized images.
    directory : str
        Directory in which the required input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    if size not in (48, 96):
        ValueError("TFD size need to be 48 or 96")

    input_file = TFD_INPUT_FILE.format(size, size)
    input_file = os.path.join(directory, input_file)
    if not os.path.isfile(input_file):
        raise MissingInputFiles('Required files missing', [input_file])

    output_file = output_file.format(size)

    with h5py.File(output_file, mode="w") as h5file:
        tfd = loadmat(input_file)

        folds = tfd['folds']
        features = tfd['images'].reshape([-1, 1, size, size])
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
    subparser.add_argument(
        "size", type=int, choices=(48, 96),
        help="height/width of the datapoints")
    subparser.set_defaults(
        func=convert_tfd,
        output_file=os.path.join(os.getcwd(), 'toronto_face_database{}.hdf5'))
