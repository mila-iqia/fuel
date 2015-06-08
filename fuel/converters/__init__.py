"""Data conversion modules for built-in datasets.

Conversion submodules generate an HDF5 file that is compatible with
their corresponding built-in dataset.

Conversion functions accept a single argument, `subparser`, which is an
`argparse.ArgumentParser` instance that it needs to fill with its own
specific arguments. They should set a `func` default argument for the
subparser with a function that will get called and given the parsed
command-line arguments, and is expected to download the required files.

"""
from fuel.converters import binarized_mnist
from fuel.converters import caltech101_silhouettes
from fuel.converters import cifar10
from fuel.converters import cifar100
from fuel.converters import mnist
from fuel.converters import svhn

__version__ = '0.2'
all_converters = (
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('caltech101_silhouettes', caltech101_silhouettes.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('cifar100', cifar100.fill_subparser),
    ('mnist', mnist.fill_subparser),
    ('svhn', svhn.fill_subparser))
