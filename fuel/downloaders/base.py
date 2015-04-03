import os
import shutil

import certifi
import urllib3
from urllib3.util.url import parse_url


def filename_from_url(url, path=None):
    """Parses a URL to determine a file name.

    Parameters
    ----------
    url : str
        URL to parse.

    """
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    with http.request('GET', url, preload_content=False) as response:
        headers = response.getheaders()
        if 'Content-Disposition' in headers:
            filename = headers[
                'Content-Disposition'].split('filename=')[1].trim('"')
        else:
            filename = os.path.basename(parse_url(url).path)
    return filename


def download(url, file_handle):
    """Downloads a given URL to a specific file.

    Parameters
    ----------
    url : str
        URL to download.
    file_handle : file
        Where to save the downloaded URL.

    """
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    with http.request('GET', url, preload_content=False) as response:
        shutil.copyfileobj(response, file_handle)


def default_downloader(args):
    """Downloads or clears files from URLs and filenames.

    This function takes an :class:`argparse.Namespace` instance as
    argument and expects it to contain three attributes:

    * `directory` : directory in which downloaded files are saved
    * `urls` : list of URLs to download
    * `filenames` : list of file names for the corresponding URLs

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed command line arguments

    """
    urls = args.urls
    save_directory = args.directory
    filenames = args.filenames

    # Parse file names from URL if not provided
    for i, url in enumerate(urls):
        filename = filenames[i]
        if not filename:
            filename = filename_from_url(url)
        if not filename:
            raise ValueError("no filename available for URL '{}'".format(url))
        filenames[i] = filename
    files = [os.path.join(save_directory, f) for f in filenames]

    if args.clear:
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
    else:
        for url, f in zip(urls, files):
            with open(f, 'wb') as file_handle:
                download(url, file_handle)
