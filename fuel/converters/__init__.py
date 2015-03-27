"""Data conversion modules for built-in datasets.

Conversion submodules generate an HDF5 file that is compatible with
their corresponding built-in dataset.

Conversion methods accept two arguments:

* `directory` : The directory containing input files expected by the
                conversion method. If not specified, `convert` should
                use the directory in which the corresponding built-in
                dataset expects to find the data.
* `save_path` : Where to save the converted data. If not specified,
                `convert` should provide a sensible default name and
                put the file in the directory in which the corresponding
                built-in dataset expects to find the data.

"""
from fuel.converters.binarized_mnist import binarized_mnist

__all__ = [binarized_mnist]
