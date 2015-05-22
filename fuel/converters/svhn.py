import os

import h5py
from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file, check_exists


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
    raise NotImplementedError(
        'SVHN format 1 conversion is not suppported at the moment.')


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
