"""Download modules for built-in datasets.

Download functions accept two arguments:

* `save_directory` : Where to save the downloaded files
* `clear` : If `True`, clear the downloaded files. Defaults to `False`.

"""
from fuel.downloaders import adult
from fuel.downloaders import binarized_mnist
from fuel.downloaders import caltech101_silhouettes
from fuel.downloaders import cifar10
from fuel.downloaders import cifar100
from fuel.downloaders import iris
from fuel.downloaders import mnist
from fuel.downloaders import svhn
from fuel.downloaders import ilsvrc2010

all_downloaders = (
    ('adult', adult.fill_subparser),
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('caltech101_silhouettes', caltech101_silhouettes.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('cifar100', cifar100.fill_subparser),
    ('iris', iris.fill_subparser),
    ('mnist', mnist.fill_subparser),
    ('svhn', svhn.fill_subparser),
    ('ilsvrc2010', ilsvrc2010.fill_subparser))
