import os
import sys
from contextlib import contextmanager
from six import wraps

import numpy
from progressbar import (ProgressBar, Percentage, Bar, ETA)

from fuel.datasets import H5PYDataset
from ..exceptions import MissingInputFiles


def check_exists(required_files):
    """Decorator that checks if required files exist before running.

    Parameters
    ----------
    required_files : list of str
        A list of strings indicating the filenames of regular files
        (not directories) that should be found in the input directory
        (which is the first argument to the wrapped function).

    Returns
    -------
    wrapper : function
        A function that takes a function and returns a wrapped function.
        The function returned by `wrapper` will include input file
        existence verification.

    Notes
    -----
    Assumes that the directory in which to find the input files is
    provided as the first argument, with the argument name `directory`.

    """
    def function_wrapper(f):
        @wraps(f)
        def wrapped(directory, *args, **kwargs):
            missing = []
            for filename in required_files:
                if not os.path.isfile(os.path.join(directory, filename)):
                    missing.append(filename)
            if len(missing) > 0:
                raise MissingInputFiles('Required files missing', missing)
            return f(directory, *args, **kwargs)
        return wrapped
    return function_wrapper


def fill_hdf5_file(h5file, data):
    """Fills an HDF5 file in a H5PYDataset-compatible manner.

    Parameters
    ----------
    h5file : :class:`h5py.File`
        File handle for an HDF5 file.
    data : tuple of tuple
        One element per split/source pair. Each element consists of a
        tuple of (split_name, source_name, data_array, comment), where

        * 'split_name' is a string identifier for the split name
        * 'source_name' is a string identifier for the source name
        * 'data_array' is a :class:`numpy.ndarray` containing the data
          for this split/source pair
        * 'comment' is a comment string for the split/source pair

        The 'comment' element can optionally be omitted.

    """
    # Check that all sources for a split have the same length
    split_names = set(split_tuple[0] for split_tuple in data)
    for name in split_names:
        lengths = [len(split_tuple[2]) for split_tuple in data
                   if split_tuple[0] == name]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("split '{}' has sources that ".format(name) +
                             "vary in length")

    # Initialize split dictionary
    split_dict = dict([(split_name, {}) for split_name in split_names])

    # Compute total source lengths and check that splits have the same dtype
    # across a source
    source_names = set(split_tuple[1] for split_tuple in data)
    for name in source_names:
        splits = [s for s in data if s[1] == name]
        indices = numpy.cumsum([0] + [len(s[2]) for s in splits])
        if not all(s[2].dtype == splits[0][2].dtype for s in splits):
            raise ValueError("source '{}' has splits that ".format(name) +
                             "vary in dtype")
        if not all(s[2].shape[1:] == splits[0][2].shape[1:] for s in splits):
            raise ValueError("source '{}' has splits that ".format(name) +
                             "vary in shapes")
        dataset = h5file.create_dataset(
            name, (sum(len(s[2]) for s in splits),) + splits[0][2].shape[1:],
            dtype=splits[0][2].dtype)
        dataset[...] = numpy.concatenate([s[2] for s in splits], axis=0)
        for i, j, s in zip(indices[:-1], indices[1:], splits):
            if len(s) == 4:
                split_dict[s[0]][name] = (i, j, None, s[3])
            else:
                split_dict[s[0]][name] = (i, j)
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)


@contextmanager
def progress_bar(name, maxval, prefix='Converting'):
    """Manages a progress bar for a conversion.

    Parameters
    ----------
    name : str
        Name of the file being converted.
    maxval : int
        Total number of steps for the conversion.

    """
    widgets = ['{} {}: '.format(prefix, name), Percentage(), ' ',
               Bar(marker='=', left='[', right=']'), ' ', ETA()]
    bar = ProgressBar(widgets=widgets, max_value=maxval, fd=sys.stdout).start()
    try:
        yield bar
    finally:
        bar.update(maxval)
        bar.finish()
