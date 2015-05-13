"""Download modules for built-in datasets.

Download functions accept two arguments:

* `save_directory` : Where to save the downloaded files
* `clear` : If `True`, clear the downloaded files. Defaults to `False`.

"""
from fuel.downloaders import binarized_mnist
from fuel.downloaders import cifar10
from fuel.downloaders import mnist

all_downloaders = (
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('mnist', mnist.fill_subparser))
