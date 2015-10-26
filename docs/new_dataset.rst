Contributing a dataset to Fuel
==============================

This tutorial describes what you need to implement in order to contribute a new
dataset to Fuel.

You need to implement the following:

* Code that downloads the raw data files for your dataset
* Code that converts these raw data files into a format that's useable by your
  dataset subclass
* Dataset subclass that interfaces with your converted data

We'll cover the basics for the following use case:

* The data consists of several data sources (*e.g.* features, targets) that
  can be stored in :class:`numpy.ndarray`-like objects
* Data sources have a fixed shape (*e.g.* vectors of size 100, images of width
  32, weight 32 and with 3 channels)
* The data is split into various sets (*e.g.* training, validation, test)

Toy example
-----------

For this tutorial, we'll implement the venerable `Iris dataset`_ in Fuel. This
dataset features 150 examples split into three classes (50 examples per class),
and each example consists of four features.

For the purpose of demonstration, we'll split the dataset into a training set
(100 examples), a validation set (20 examples) and a test set (30 examples).
We'll pretend the test set doesn't have any label information available,
as is often the case for machine learning competitions.

Download code
-------------

The Iris dataset is contained in a single file, `iris.data`_, which we'll need
to make available for users to download.

The preferred way of downloading dataset files in Fuel is the ``fuel-download``
script. Dataset implementations include a function for downloading their required
files in the ``fuel.downloaders`` subpackage. In order to make that function
accessible to ``fuel-download``, they need to include it in the
``all_downloaders`` attribute of the ``fuel.downloaders`` subpackage.

The function accepts an :class:`argparse.ArgumentParser` instance as input and
should return a downloading function. Put the following piece of code inside
the ``fuel.downloaders.iris`` module (you'll have to create it):

.. code-block:: python

    from fuel.downloaders.base import default_downloader

    def fill_subparser(subparser):
        subparser.set_defaults(
            urls=['https://archive.ics.uci.edu/ml/machine-learning-databases/'
                  'iris/iris.data'],
            filenames=['iris.data'])
        return default_downloader

You should also register Iris as a downloadable dataset via the
``all_downloaders`` attribute. It's a tuple of pairs of name and subparser
filler function. Here's an example of how the ``fuel.downloaders`` init file
might look:

.. code-block:: python

    from fuel.downloaders import binarized_mnist
    from fuel.downloaders import iris

    all_downloaders = (
        ('binarized_mnist', binarized_mnist.fill_subparser),
        ('iris', iris.fill_subparser))

A lot is going on in these few lines of code, so let's break it down.

In order to be more flexible, the ``fuel-download`` script uses subparsers.
This lets each dataset define their own set of arguments. If you registered it
properly, the function you just defined will get called and be given its own
subparser to fill. Users will then be able to type the ``fuel-download iris``
command and ``iris.data`` will be downloaded.

When the ``fuel-download iris`` command is typed, the download script will call
the function returned by ``fill_subparser`` and give it the
:class:`argparse.Namespace` instance containing all parsed command line
arguments. That function is responsible for downloading the data.

We used the :meth:`~.downloaders.base.default_downloader` convenience function
as our download function. It expects the parsed arguments to contain a list of
URLs and a list of filenames, and downloads each URL, saving it under its
corresponding filename. This is why we set the ``urls`` and ``filenames``
default arguments.

If your use case is more exotic, you can just as well define your own download
function. Be aware of the following default arguments:

* ``directory`` : in which directory the files need to be saved
* ``clear`` : if ``True``, your download function is expected to remove the
  downloaded files from ``directory``.

Conversion code
---------------

In order to minimize the amount of code we have to write, we'll subclass
:class:`~.datasets.hdf5.H5PYDataset`. This means we'll have to create an HDF5
file to store our data. For more information, see the :ref:`dedicated tutorial
<convert_h5py_dataset>` on how to create an
:class:`~.datasets.hdf5.H5PYDataset`-compatible HDF5 file.

Much like for downloading data files, the preferred way of converting data
files in Fuel is through the ``fuel-convert`` script. Its implementation is
very similar to ``fuel-download``. The arguments to be aware of in the subparser
are

* ``directory`` : in which directory the input files reside
* ``output-directory`` : where to save the converted dataset

The converter function should return a tuple containing path(s) to the converted
dataset(s).

Put the following piece of code inside the ``fuel.converters.iris``
module (you'll have to create it):


.. code-block:: python

    import os

    import h5py
    import numpy

    from fuel.converters.base import fill_hdf5_file


    def convert_iris(directory, output_directory, output_filename='iris.hdf5'):
        output_path = os.path.join(output_directory, output_filename)
        h5file = h5py.File(output_path, mode='w')
        classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        data = numpy.loadtxt(
            os.path.join(directory, 'iris.data'),
            converters={4: lambda x: classes[x]},
            delimiter=',')
        numpy.random.shuffle(data)
        features = data[:, :-1].astype('float32')
        targets = data[:, -1].astype('uint8')
        train_features = features[:100]
        train_targets = targets[:100]
        valid_features = features[100:120]
        valid_targets = targets[100:120]
        test_features = features[120:]
        data = (('train', 'features', train_features),
                ('train', 'targets', train_targets),
                ('valid', 'features', valid_features),
                ('valid', 'targets', valid_targets),
                ('test', 'features', test_features))
        fill_hdf5_file(h5file, data)
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'

        h5file.flush()
        h5file.close()

        return (output_path,)

    def fill_subparser(subparser):
        return convert_iris

We used the convenience :meth:`~.converters.base.fill_hdf5_file` function
to populate our HDF5 file and create the split array. This function expects
a tuple of tuples, one per split/source pair, containing the split name,
the source name, the data array and (optionally) a comment string.

We also used :class:`~.datasets.hdf5.H5PYDataset`'s ability to extract axis
labels to add semantic information to the axes of our data sources. This
allowed us to specify that target values are categorical (``'index``'). Note
that you can use whatever label you want in Fuel, although certain frameworks
using Fuel may have some hard-coded assumptions about which labels to use.

As for the download code, you should register Iris as a convertible dataset
via the ``all_converters`` attribute of the ``fuel.converters`` subpackage.
Here's an example of how the init file might look:

.. code-block:: python

    from fuel.converters import binarized_mnist
    from fuel.converters import iris

    all_converters = (
        ('binarized_mnist', binarized_mnist.fill_subparser),
        ('iris', iris.fill_subparser))

Dataset subclass
----------------

Let's now implement the :class:`~.datasets.hdf5.H5PYDataset` subclass that will
interface with our newly-created HDF5 file.

One advantage of subclassing :class:`~.datasets.hdf5.H5PYDataset` is that the
amount of code to write is very minimal:

.. code-block:: python

    from fuel.datasets import H5PYDataset
    from fuel.utils import find_in_data_path


    class Iris(H5PYDataset):
        filename = 'iris.hdf5'

        def __init__(self, which_sets=which_sets, **kwargs):
            kwargs.setdefault('load_in_memory', True)
            super(Iris, self).__init__(
                file_or_path=find_in_data_path(self.filename),
                which_sets=which_sets, **kwargs)

Our subclass is just a thin wrapper around the
:class:`~.datasets.hdf5.H5PYDataset` class that defines the data path and
switches the ``load_in_memory`` argument default to ``True`` (since this dataset
easily fits in memory). Everything else is handled by the superclass.

Putting it together
-------------------

We now have everything we need to start playing around with our new dataset
implementation.

Try downloading and converting the data file:

.. code-block:: bash

    cd $FUEL_DATA_PATH
    fuel-download iris
    fuel-convert iris
    fuel-download iris --clear
    cd -

You can now use the Iris dataset like you would use any other built-in dataset:

.. doctest::
    :hide:
    
    >>> import mock
    >>> import os
    >>> from picklable_itertools import chain
    >>> from six.moves import range
    >>> from fuel.downloaders.base import default_downloader
    >>> def fill_downloader_subparser(subparser):
    ...     subparser.set_defaults(
    ...         urls=['https://archive.ics.uci.edu/ml/machine-learning-databases/'
    ...               'iris/iris.data'],
    ...         filenames=['iris.data'])
    ...     return default_downloader
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> __ = parser.add_argument("--directory", type=str, default=os.getcwd())
    >>> __ = parser.add_argument("--clear", action='store_true')
    >>> subparsers = parser.add_subparsers()
    >>> download_function = fill_downloader_subparser(subparsers.add_parser('iris'))
    >>> args = parser.parse_args(['iris'])
    >>> args_dict = vars(args)
    >>> content = b''
    >>> for i in range(50):
    ...    content += b'0.0,0.0,0.0,0.0,Iris-setosa\n'
    >>> for i in range(50):
    ...    content += b'0.0,0.0,0.0,0.0,Iris-versicolor\n'
    >>> for i in range(50):
    ...    content += b'0.0,0.0,0.0,0.0,Iris-virginica\n'
    >>> length = len(content)
    >>> @mock.patch('fuel.downloaders.base.requests')
    ... def call_download(func, args_dict, mock_requests):
    ...     mock_response = mock.Mock()
    ...     mock_response.iter_content = mock.Mock(
    ...         side_effect = lambda s: chain(
    ...             (content[s * i: s * (i + 1)] for i in range(length // s)),
    ...             (content[(length // s) * s:],)))
    ...     mock_response.headers = {'content-length': length}
    ...     mock_requests.get.return_value = mock_response
    ...     func(**args_dict)
    >>> call_download(download_function, args_dict) # doctest: +ELLIPSIS
    Downloading ...

.. doctest::
    :hide:
    
    >>> import h5py
    >>> import numpy
    >>> from fuel.converters.base import fill_hdf5_file
    >>> def iris_converter(directory, output_directory, output_filename='iris.hdf5'):
    ...     output_path = os.path.join(output_directory, output_filename)
    ...     h5file = h5py.File(output_path, mode='w')
    ...     classes = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    ...     data = numpy.loadtxt(
    ...         os.path.join(directory, 'iris.data'),
    ...         converters={4: lambda x: classes[x]},
    ...         delimiter=',')
    ...     numpy.random.shuffle(data)
    ...     features = data[:, :-1].astype('float32')
    ...     targets = data[:, -1:].astype('uint8').reshape((-1, 1))
    ...     train_features = features[:100]
    ...     train_targets = targets[:100]
    ...     valid_features = features[100:120]
    ...     valid_targets = targets[100:120]
    ...     test_features = features[120:]
    ...     data = (('train', 'features', train_features),
    ...             ('train', 'targets', train_targets),
    ...             ('valid', 'features', valid_features),
    ...             ('valid', 'targets', valid_targets),
    ...             ('test', 'features', test_features))
    ...     fill_hdf5_file(h5file, data)
    ...     h5file['features'].dims[0].label = 'batch'
    ...     h5file['features'].dims[1].label = 'feature'
    ...     h5file['targets'].dims[0].label = 'batch'
    ...     h5file['targets'].dims[1].label = 'index'
    ...     h5file.flush()
    ...     h5file.close()
    ...     return (output_path,)
    >>> def fill_converter_subparser(subparser):
    ...     return iris_converter
    >>> parser = argparse.ArgumentParser()
    >>> __ = parser.add_argument("--directory", type=str, default=os.getcwd())
    >>> __ = parser.add_argument("--output-directory", type=str,
    ...                          default=os.getcwd())
    >>> subparsers = parser.add_subparsers()
    >>> convert_function = fill_converter_subparser(subparsers.add_parser('iris'))
    >>> args = parser.parse_args(['iris'])
    >>> args_dict = vars(args)
    >>> output_paths = convert_function(**args_dict)
    >>> os.remove('iris.data')

.. doctest::
    :hide:
    
    >>> import os
    >>> from fuel import config
    >>> from fuel.datasets import H5PYDataset
    >>> class Iris(H5PYDataset):
    ...    def __init__(self, which_sets, **kwargs):
    ...        kwargs.setdefault('load_in_memory', True)
    ...        super(Iris, self).__init__('iris.hdf5', which_sets, **kwargs)

.. doctest::

    >>> from fuel.datasets.iris import Iris # doctest: +SKIP
    >>> train_set = Iris(('train',))
    >>> print(train_set.axis_labels['features'])
    ('batch', 'feature')
    >>> print(train_set.axis_labels['targets'])
    ('batch', 'index')
    >>> handle = train_set.open()
    >>> data = train_set.get_data(handle, slice(0, 10))
    >>> print((data[0].shape, data[1].shape))
    ((10, 4), (10, 1))
    >>> train_set.close(handle)

.. doctest::
    :hide:
    
    >>> os.remove('iris.hdf5')

Working with external packages
------------------------------

To distribute Fuel-compatible downloaders and converters independently from
Fuel, you have to write your own modules or subpackages which will define
``all_downloaders`` and ``all_converters``. Here is how the Iris downloader
and converter might look like if you were to include them as part of the
``my_fuel`` package:

.. code-block:: python

    # my_fuel/downloaders/iris_downloader.py
    from fuel.downloaders.base import default_downloader

    def fill_subparser(subparser):
        subparser.set_defaults(
            urls=['https://archive.ics.uci.edu/ml/machine-learning-databases/'
                  'iris/iris.data'],
            filenames=['iris.data'])
        return default_downloader

.. code-block:: python

    # my_fuel/downloaders/__init__.py
    from my_fuel.downloaders import iris

    all_downloaders = (('iris', iris.fill_subparser),)

.. code-block:: python

    # my_fuel/converters/iris.py
    import os

    import h5py
    import numpy

    from fuel.converters.base import fill_hdf5_file


    def convert_iris(directory, output_directory, output_filename='iris.hdf5'):
        output_path = os.path.join(output_directory, output_filename)
        h5file = h5py.File(output_path, mode='w')
        classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        # ...

    def fill_subparser(subparser):
        return convert_iris

.. code-block:: python

    # my_fuel/converters/__init__.py
    from my_fuel.converters import iris

    all_converters = (('iris', iris.fill_subparser),)

In order to use the downloaders and converters defined in ``my_fuel``, users
would then have to set the ``extra_downloaders`` and ``extra_converters``
configuration variables inside ``~/.fuelrc`` like so:

.. code-block:: yaml

    extra_downloaders: ['my_fuel.downloaders']
    extra_converters: ['my_fuel.converters']

or set the ``FUEL_EXTRA_DOWNLOADERS`` and ``FUEL_EXTRA_CONVERTERS`` environment
variables like so (this would override the ``extra_downloaders`` and
``extra_converters`` configuration variables):

.. code-block:: bash

    $ export FUEL_EXTRA_DOWNLOADERS=my_fuel.downloaders
    $ export FUEL_EXTRA_CONVERTERS=my_fuel.converters

.. _Iris dataset: https://archive.ics.uci.edu/ml/datasets/Iris
.. _iris.data: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
