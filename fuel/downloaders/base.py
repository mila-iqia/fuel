import os
import re
import sys
from contextlib import contextmanager

import requests
from progressbar import (ProgressBar, Percentage, Bar, ETA, FileTransferSpeed,
                         Timer, UnknownLength)
from six.moves import zip, urllib
from ..exceptions import NeedURLPrefix


@contextmanager
def progress_bar(name, maxval):
    """Manages a progress bar for a download.

    Parameters
    ----------
    name : str
        Name of the downloaded file.
    maxval : int
        Total size of the download, in bytes.

    """
    if maxval is not UnknownLength:
        widgets = ['{}: '.format(name), Percentage(), ' ',
                   Bar(marker='=', left='[', right=']'), ' ', ETA(), ' ',
                   FileTransferSpeed()]
    else:
        widgets = ['{}: '.format(name), ' ', Timer(), ' ', FileTransferSpeed()]
    bar = ProgressBar(widgets=widgets, max_value=maxval, fd=sys.stdout).start()
    try:
        yield bar
    finally:
        bar.update(maxval)
        bar.finish()


def filename_from_url(url, path=None):
    """Parses a URL to determine a file name.

    Parameters
    ----------
    url : str
        URL to parse.

    """
    r = requests.get(url, stream=True)
    if 'Content-Disposition' in r.headers:
        filename = re.findall(r'filename=([^;]+)',
                              r.headers['Content-Disposition'])[0].strip('"\"')
    else:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    return filename


def download(url, file_handle, chunk_size=1024):
    """Downloads a given URL to a specific file.

    Parameters
    ----------
    url : str
        URL to download.
    file_handle : file
        Where to save the downloaded URL.

    """
    r = requests.get(url, stream=True)
    total_length = r.headers.get('content-length')
    if total_length is None:
        maxval = UnknownLength
    else:
        maxval = int(total_length)
    name = file_handle.name
    with progress_bar(name=name, maxval=maxval) as bar:
        for i, chunk in enumerate(r.iter_content(chunk_size)):
            if total_length:
                bar.update(i * chunk_size)
            file_handle.write(chunk)


def ensure_directory_exists(directory):
    """Create directory (with parents) if does not exist, raise on failure.

    Parameters
    ----------
    directory : str
        The directory to create

    """
    if os.path.isdir(directory):
        return
    os.makedirs(directory)


def default_downloader(directory, urls, filenames, url_prefix=None,
                       clear=False):
    """Downloads or clears files from URLs and filenames.

    Parameters
    ----------
    directory : str
        The directory in which downloaded files are saved.
    urls : list
        A list of URLs to download.
    filenames : list
        A list of file names for the corresponding URLs.
    url_prefix : str, optional
        If provided, this is prepended to filenames that
        lack a corresponding URL.
    clear : bool, optional
        If `True`, delete the given filenames from the given
        directory rather than download them.

    """
    # Parse file names from URL if not provided
    for i, url in enumerate(urls):
        filename = filenames[i]
        if not filename:
            filename = filename_from_url(url)
        if not filename:
            raise ValueError("no filename available for URL '{}'".format(url))
        filenames[i] = filename
    files = [os.path.join(directory, f) for f in filenames]

    if clear:
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
    else:
        print('Downloading ' + ', '.join(filenames) + '\n')
        ensure_directory_exists(directory)

        for url, f, n in zip(urls, files, filenames):
            if not url:
                if url_prefix is None:
                    raise NeedURLPrefix
                url = url_prefix + n
            with open(f, 'wb') as file_handle:
                download(url, file_handle)
