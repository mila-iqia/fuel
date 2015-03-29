import os
import shutil

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
    http = urllib3.PoolManager()
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
