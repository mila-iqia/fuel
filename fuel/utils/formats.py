"""Low-level utilities for reading a variety of source formats."""
import tarfile
import six


def tar_open(f):
    """Open either a filename or a file-like object as a TarFile.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-like object from which to read.

    Returns
    -------
    TarFile
        A `TarFile` instance.

    """
    if isinstance(f, six.string_types):
        return tarfile.open(name=f)
    else:
        return tarfile.open(fileobj=f)
