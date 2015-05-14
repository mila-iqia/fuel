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
from fuel.converters import cifar10
from fuel.converters import mnist

__version__ = '0.1'
all_converters = (
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('mnist', mnist.fill_subparser))
