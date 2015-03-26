# flake8: noqa
from fuel.datasets.base import (Dataset, IterableDataset,
                                IndexableDataset)

from fuel.datasets.hdf5 import H5PYDataset
from fuel.datasets.binarized_mnist import BinarizedMNIST
from fuel.datasets.cifar10 import CIFAR10
from fuel.datasets.mnist import MNIST
from fuel.datasets.text import TextFile
from fuel.datasets.billion import OneBillionWord
