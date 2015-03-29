import os
import shutil

import certifi
import urllib3
from urllib3.util.url import parse_url


def download(url, path=None):
    """Downloads a given URL to a specific directory or file.

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Where to save the downloaded URL. Defaults to `None`, in which
        case the current working directory is used.

    """
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    with http.request('GET', url, preload_content=False) as response:
        if not path:
            path = os.getcwd()
        if os.path.isdir(path):
            headers = response.getheaders()
            if 'Content-Disposition' in headers:
                filename = headers[
                    'Content-Disposition'].split('filename=')[1].trim('"')
            else:
                filename = os.path.basename(parse_url(url).path)
            if not filename:
                raise ValueError("no filename given")
            path = os.path.join(path, filename)
        with open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)


def default_manager(urls, filenames):
    """Returns a function to download or clear files from a list of URLs.

    Parameters
    ----------
    urls : list of str
        URLs to download from.
    filenames : list of str
        Files to save to.

    """
    def manager_fn(args):
        save_directory = args.directory
        files = [os.path.join(save_directory, f) for f in filenames]
        if args.clear:
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)
        else:
            for url, f in zip(urls, files):
                download(url, f)
    return manager_fn
