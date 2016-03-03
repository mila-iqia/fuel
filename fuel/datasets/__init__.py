# flake8: noqa
from fuel.datasets.base import (Dataset, IterableDataset,
                                IndexableDataset)

from fuel.datasets.hdf5 import H5PYDataset
from fuel.datasets.adult import Adult
from fuel.datasets.binarized_mnist import BinarizedMNIST
from fuel.datasets.celeba import CelebA
from fuel.datasets.cifar10 import CIFAR10
from fuel.datasets.cifar100 import CIFAR100
from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.datasets.iris import Iris
from fuel.datasets.mnist import MNIST
from fuel.datasets.svhn import SVHN
from fuel.datasets.text import TextFile
from fuel.datasets.billion import OneBillionWord
