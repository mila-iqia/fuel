"""Data conversion modules for built-in datasets.

Conversion submodules generate an HDF5 file that is compatible with
their corresponding built-in dataset.

Conversion functions accept a single argument, `subparser`, which is an
`argparse.ArgumentParser` instance that it needs to fill with its own
specific arguments. They should set a `func` default argument for the
subparser with a function that will get called and given the parsed
command-line arguments, and is expected to download the required files.

"""
from fuel.converters import adult
from fuel.converters import binarized_mnist
from fuel.converters import caltech101_silhouettes
from fuel.converters import celeba
from fuel.converters import cifar10
from fuel.converters import cifar100
from fuel.converters import dogs_vs_cats
from fuel.converters import iris
from fuel.converters import mnist
from fuel.converters import svhn
from fuel.converters import ilsvrc2010
from fuel.converters import ilsvrc2012
from fuel.converters import youtube_audio

__version__ = '0.2'
all_converters = (
    ('adult', adult.fill_subparser),
    ('binarized_mnist', binarized_mnist.fill_subparser),
    ('caltech101_silhouettes', caltech101_silhouettes.fill_subparser),
    ('celeba', celeba.fill_subparser),
    ('cifar10', cifar10.fill_subparser),
    ('cifar100', cifar100.fill_subparser),
    ('dogs_vs_cats', dogs_vs_cats.fill_subparser),
    ('iris', iris.fill_subparser),
    ('mnist', mnist.fill_subparser),
    ('svhn', svhn.fill_subparser),
    ('ilsvrc2010', ilsvrc2010.fill_subparser),
    ('ilsvrc2012', ilsvrc2012.fill_subparser),
    ('youtube_audio', youtube_audio.fill_subparser))
