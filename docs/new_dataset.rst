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
accessible to ``fuel-download``, they need to include it in the ``__all__``
attribute of the ``fuel.downloaders`` subpackage.

The function accepts an :class:`argparse.ArgumentParser` instance as input and
should set a function as the default value for the ``func`` argument of the
parser. Put the following piece of code inside the ``fuel.downloaders.iris``
module (you'll have to create it):

.. code-block:: python

    from fuel.downloaders.base import default_downloader

    def iris(subparser):
        subparser.set_defaults(
            func=default_downloader,
            urls=['https://archive.ics.uci.edu/ml/machine-learning-databases/'
                  'iris/iris.data'],
            filenames=['iris.data'])

You should also import the function you just defined and add ``'iris'`` inside
the ``__all__`` attribute of the ``fuel.downladers`` init file. Here's an
example of how the init file might look:

.. code-block:: python

    from fuel.downloaders.binarized_mnist import binarized_mnist
    from fuel.downloaders.iris import iris

    __all__ = ('binarized_mnist', 'iris')

A lot is going on in these few lines of code, so let's break it down.

In order to be more flexible, the ``fuel-download`` script uses subparsers.
This lets each dataset define their own set of arguments. If you registered it
properly, the function you just defined will get called and be given its own
subparser to fill. Users will then be able to type the ``fuel-download iris``
command and ``iris.data`` will be downloaded.

When the ``fuel-download iris`` command is typed, the download script will call
the function passed as the ``func`` argument and give it the
:class:`argparse.Namespace` instance containing all parsed command line
arguments. That function is responsible for downloading the data.

This is why we called the ``set_defaults`` method: it allowed us to define which
function would get called to download our data. We used the
:meth:`~.downloaders.base.default_downloader` convenience function. It expects
the parsed arguments to contain a list of URLs and a list of filenames,
and downloads each URL, saving it under its corresponding filename. This is why
we also included the ``urls`` and ``filenames`` default arguments.

If your use case is more exotic, you can just as well define your own download
function. Be aware of the following parser-level arguments:

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
somewhat simpler: instead of registering a function that fills a subparser,
you register a function that takes an input directory path and an output file
path as argument. It looks for its required files in the input directory path
and saves the converted data at the output file path.

Put the following piece of code inside the ``fuel.converters.iris``
module (you'll have to create it):


.. code-block:: python

    import os

    import h5py
    import numpy

    from fuel.converters.base import fill_hdf5_file


    def iris(input_directory, save_path):
        h5file = h5py.File(save_path, mode="w")
        classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        data = numpy.loadtxt(
            os.path.join(input_directory, 'iris.data'),
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

We used the convenience :meth:`~.converters.base.fill_hdf5_file` function
to populate our HDF5 file and create the split array. This function expects
a tuple of tuples, one per split/source pair, containing the split name,
the source name, the data array and (optionally) a comment string.

We also used :class:`~.datasets.hdf5.H5PYDataset`'s ability to extract axis
labels to add semantic information to the axes of our data sources. This
allowed us to specify that target values are categorical (``'index``'). Note
that you can use whatever label you want in Fuel, although certain frameworks
using Fuel may have some hard-coded assumptions about which labels to use.

As for the download code, you should import the function you just defined and
add ``'iris'`` inside the ``__all__`` attribute of the ``fuel.converters`` init
file. Here's an example of how the init file might look:

.. code-block:: python

    from fuel.converters.binarized_mnist import binarized_mnist
    from fuel.converters.iris import iris

    __all__ = ('binarized_mnist', 'iris')

Dataset subclass
----------------

Let's now implement the :class:`~.datasets.hdf5.H5PYDataset` subclass that will
interface with our newly-created HDF5 file.

One advantage of subclassing :class:`~.datasets.hdf5.H5PYDataset` is that the
amount of code to write is very minimal:

.. code-block:: python

    import os

    from fuel import config
    from fuel.datasets import H5PYDataset


    class Iris(H5PYDataset):
        filename = 'iris.hdf5'

        def __init__(self, which_set, **kwargs):
            kwargs.setdefault('load_in_memory', True)
            super(Iris, self).__init__(self.data_path, which_set, **kwargs)

        @property
        def data_path(self):
            return os.path.join(config.data_path, self.filename)

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
    fuel-download --clear iris
    cd -

You can now use the Iris dataset like you would use any other built-in dataset:

.. code-block:: python

    >>> from fuel.datasets.iris import Iris # doctest: +SKIP
    >>> train_set = Iris('train') # doctest: +SKIP
    >>> print(train_set.axis_labels) # doctest: +SKIP
    {'features': ('batch', 'feature'), 'targets': ('batch', 'index')}
    >>> handle = train_set.open() # doctest: +SKIP
    >>> data = train_set.get_data(handle, slice(0, 10)) # doctest: +SKIP
    >>> print((data[0].shape, data[1].shape)) # doctest: +SKIP
    ((10, 4), (10,))
    >>> train_set.close(handle) # doctest: +SKIP

.. _Iris dataset: https://archive.ics.uci.edu/ml/datasets/Iris
.. _iris.data: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
