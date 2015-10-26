import os

import h5py
import numpy

from fuel.converters.base import fill_hdf5_file


def convert_iris(directory, output_directory, output_filename='iris.hdf5'):
    """Convert the Iris dataset to HDF5.

    Converts the Iris dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.Iris`. The converted dataset is
    saved as 'iris.hdf5'.
    This method assumes the existence of the file `iris.data`.

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to `None`, in which case a name
        based on `dtype` will be used.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    classes = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    data = numpy.loadtxt(
        os.path.join(directory, 'iris.data'),
        converters={4: lambda x: classes[x]},
        delimiter=',')
    features = data[:, :-1].astype('float32')
    targets = data[:, -1].astype('uint8').reshape((-1, 1))
    data = (('all', 'features', features),
            ('all', 'targets', targets))

    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'feature'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the Iris dataset file.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `iris` command.

    """
    return convert_iris
