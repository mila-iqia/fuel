"""Download modules for built-in datasets.

Download methods accept a single argument, `save_path`, which tells
where to save the downloaded files.

"""
from fuel.downloaders.binarized_mnist import binarized_mnist

__all__ = ('binarized_mnist',)
