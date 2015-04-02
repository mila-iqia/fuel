import numpy


def fill_hdf5_file(h5file, data, source_names, shapes, dtypes,
                   split_names, splits):
    """Fills an HDF5 file in a H5PYDataset-compatible manner.

    Parameters
    ----------
    h5file : :class:`h5py.File`
        File handle for an HDF5 file.
    data : list of lists of :class:`numpy.ndarray`
        Rows correspond to sources, columns correspond to splits.
    source_names : tuple of str
        Source names of the corresponding rows in `data`.
    shapes : tuple of tuples of int
        Shape of the concatenated splits of the corresponding rows in `data`.
    dtypes : tuple of str
        Data types of the corresponding rows in `data`.
    split_names : tuple of str
        Split names of the corresponding columns in `data`.
    splits : tuple of :class:`numpy.ndarray`
        Split start and stop indices of the corresponding columns in `data`.

    """
    for name, split in zip(split_names, splits):
        h5file.attrs[name] = split
    for source, name, shape, dtype in zip(data, source_names, shapes, dtypes):
        dataset = h5file.create_dataset(name, shape, dtype=dtype)
        dataset[...] = numpy.vstack(source)
