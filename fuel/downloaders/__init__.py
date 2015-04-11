"""Download modules for built-in datasets.

Download functions accept two arguments:

* `save_directory` : Where to save the downloaded files
* `clear` : If `True`, clear the downloaded files. Defaults to `False`.

"""
from fuel.downloaders.binarized_mnist import binarized_mnist

__all__ = ('binarized_mnist',)
