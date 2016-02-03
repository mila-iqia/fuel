"""Low-level utilities for reading a variety of source formats."""
import codecs
import gzip
import io
import tarfile
import six


def open_(filename, mode='r', encoding=None):
    """Open a text file with encoding and optional gzip compression.

    Note that on legacy Python any encoding other than ``None`` or opening
    GZipped files will return an unpicklable file-like object.

    Parameters
    ----------
    filename : str
        The filename to read.
    mode : str, optional
        The mode with which to open the file. Defaults to `r`.
    encoding : str, optional
        The encoding to use (see the codecs documentation_ for supported
        values). Defaults to ``None``.

    .. _documentation:
    https://docs.python.org/3/library/codecs.html#standard-encodings

    """
    if filename.endswith('.gz'):
        if six.PY2:
            zf = io.BufferedReader(gzip.open(filename, mode))
            if encoding:
                return codecs.getreader(encoding)(zf)
            else:
                return zf
        else:
            return io.BufferedReader(gzip.open(filename, mode,
                                               encoding=encoding))
    if six.PY2:
        if encoding:
            return codecs.open(filename, mode, encoding=encoding)
        else:
            return open(filename, mode)
    else:
        return open(filename, mode, encoding=encoding)


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
