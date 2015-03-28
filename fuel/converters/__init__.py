"""Data conversion modules for built-in datasets.

Conversion submodules generate an HDF5 file that is compatible with
their corresponding built-in dataset.

Conversion functions accept two arguments:

* `input_directory` : Directory containing input files expected by the
                      conversion function.
* `save_directory`  : Directory in which the converted data is saved.

"""
from fuel.converters.binarized_mnist import binarized_mnist

__all__ = ('binarized_mnist',)
