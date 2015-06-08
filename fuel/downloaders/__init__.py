"""Download modules for built-in datasets.

Download functions accept two arguments:

* `save_directory` : Where to save the downloaded files
* `clear` : If `True`, clear the downloaded files. Defaults to `False`.

"""
from fuel.downloaders import binarized_mnist
from fuel.downloaders import caltech101_silhouettes
from fuel.downloaders import cifar10
from fuel.downloaders import cifar100
from fuel.downloaders import mnist
from fuel.downloaders import svhn

all_downloaders = (
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('caltech101_silhouettes', caltech101_silhouettes.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('cifar100', cifar100.fill_subparser),
    ('mnist', mnist.fill_subparser),
    ('svhn', svhn.fill_subparser))
