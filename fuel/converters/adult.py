import os

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file


def convert_to_one_hot(y):
    """
    converts y into one hot reprsentation.

    Parameters
    ----------
    y : list
        A list containing continous integer values.

    Returns
    -------
    one_hot : numpy.ndarray
        A numpy.ndarray object, which is one-hot representation of y.

    """
    max_value = max(y)
    min_value = min(y)
    length = len(y)
    one_hot = numpy.zeros((length, (max_value - min_value + 1)))
    one_hot[numpy.arange(length), y] = 1
    return one_hot


def convert_adult(directory, output_directory,
                  output_filename='adult.hdf5'):
    """
    Convert the Adult dataset to HDF5.

    Converts the Adult dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.Adult`. The converted dataset is saved as
    'adult.hdf5'.
    This method assumes the existence of the file `adult.data` and
    `adult.test`.

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to `adult.hdf5`.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    train_path = os.path.join(directory, 'adult.data')
    test_path = os.path.join(directory, 'adult.test')
    output_path = os.path.join(output_directory, output_filename)

    train_content = open(train_path, 'r').readlines()
    test_content = open(test_path, 'r').readlines()
    train_content = train_content[:-1]
    test_content = test_content[1:-1]

    features_list = []
    targets_list = []
    for content in [train_content, test_content]:
        # strip out examples with missing features
        content = [line for line in content if line.find('?') == -1]
        # strip off endlines, separate entries
        content = list(map(lambda l: l[:-1].split(', '), content))

        features = list(map(lambda l: l[:-1], content))
        targets = list(map(lambda l: l[-1], content))
        del content
        y = list(map(lambda l: [l[0] == '>'], targets))
        y = numpy.array(y)
        del targets

        # Process features into a matrix
        variables = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ]
        continuous = set([
            'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
            'hours-per-week'
        ])

        pieces = []
        for i, var in enumerate(variables):
            data = list(map(lambda l: l[i], features))
            if var in continuous:
                data = list(map(lambda l: float(l), data))
                data = numpy.array(data)
                data = data.reshape(data.shape[0], 1)
            else:
                unique_values = list(set(data))
                data = list(map(lambda l: unique_values.index(l), data))
                data = convert_to_one_hot(data)
            pieces.append(data)

        X = numpy.concatenate(pieces, axis=1)

        features_list.append(X)
        targets_list.append(y)

    # the largets value in the last variable of test set is only 40, thus
    # the one hot representation has 40 at the second dimention. While in
    # training set it is 41. Since it lies in the last variable, so it is
    # safe to simply add a last column with zeros.
    features_list[1] = numpy.concatenate(
        (features_list[1],
         numpy.zeros((features_list[1].shape[0], 1),
                     dtype=features_list[1].dtype)),
        axis=1)
    h5file = h5py.File(output_path, mode='w')
    data = (('train', 'features', features_list[0]),
            ('train', 'targets', targets_list[0]),
            ('test', 'features', features_list[1]),
            ('test', 'targets', targets_list[1]))

    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'feature'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    return convert_adult
