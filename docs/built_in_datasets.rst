Built-in datasets
=================

Fuel has a growing number of built-in datasets that simplify working on
standard benchmark datasets, such as MNIST or CIFAR10.

These datasets are defined in the ``fuel.datasets`` module. Some user
intervention is needed before they're used for the first time: a given
dataset has to be downloaded and converted into a format that is recognized by
its corresponding dataset class. Fortunately, Fuel also has built-in tools
to automate these operations.

Environment variable
--------------------

In order for Fuel to know where to look for its data, the ``data_path``
configuration variable has to be set inside ``~/.fuelrc``. It's expected to be
a sequence of paths separated by an OS-specific delimiter (``:`` for Linux and
OSX, ``;`` for Windows):

.. code-block:: yaml

    # ~/.fuelrc
    data_path: "/first/path/to/my/data:/second/path/to/my/data"

When looking for a specific file (e.g. ``mnist.hdf5``), Fuel will search each of
these paths in sequence, using the first matching file that it finds.

This configuration variable can be overridden by setting the ``FUEL_DATA_PATH``
environment variable:

.. code-block:: bash

    $ export FUEL_DATA_PATH="/first/path/to/my/data:/second/path/to/my/data"

Let's now change directory for the rest of this tutorial:

.. code-block:: bash

    $ cd $FUEL_DATA_PATH

Download a built-in dataset
---------------------------

We're going to download the raw data files for the MNIST dataset with the
``fuel-download`` script that was installed with Fuel:

.. code-block:: bash

    $ fuel-download mnist

The script is pretty simple: you call it and pass it the name of the dataset
you'd like to download. In order to know which datasets are available to
download via ``fuel-download``, type

.. code-block:: bash

    $ fuel-download -h

You can pass dataset-specific arguments to the script. In order to know which
arguments are accepted, append ``-h`` to your dataset choice:

.. code-block:: bash

    fuel-download mnist -h

Two arguments are always accepted:

* ``-d DIRECTORY`` : define where the dataset files will be downloaded. By
  default, ``fuel-download`` uses the current working directory.
* ``--clear`` : delete the dataset files instead of downloading them, if they
  exist.

Convert downloaded files
------------------------

You should now have four new files in your directory:

* ``train-images-idx3-ubyte.gz``
* ``train-labels-idx1-ubyte.gz``
* ``t10k-images-idx3-ubyte.gz``
* ``t10k-labels-idx1-ubyte.gz``

Those are the original files that can be downloaded off Yann Lecun's website.
We now need to convert those files into a format that the ``MNIST`` dataset
class will recognize. This is done through the ``fuel-convert`` script:

.. code-block:: bash

    $ fuel-convert mnist

This will generate an ``mnist.hdf5`` file in your directory, which the
``MNIST`` class recognizes.

Once again, the script accepts dataset-specific arguments which you can discover
by appending ``-h`` to your dataset choice:

.. code-block:: bash

    fuel-convert mnist -h

Two arguments are always accepted:

* ``-d DIRECTORY`` : where ``fuel-convert`` should look for the input files.
* ``-o OUTPUT_FILE`` : where to save the converted dataset.

Let's delete the raw input files, as we don't need them anymore:

.. code-block:: bash

    $ fuel-download mnist --clear

Inspect Fuel-generated dataset files
------------------------------------

Six months from now, you may have a bunch of dataset files lying on disk, each
with slight differences that you can't identify or reproduce. At that time,
you'll be glad that ``fuel-info`` exists.

When a dataset is generated through ``fuel-convert``, the script tags it with
what command was issued to generate the file and what were the versions of
relevant parts of the library at that time.

You can inspect this metadata calling ``fuel-info`` and passing an HDF5 file as
argument:

.. code-block:: bash

    $ fuel-info mnist.hdf5

.. code-block:: text

    Metadata for mnist.hdf5
    =======================

    The command used to generate this file is

        fuel-convert mnist

    Relevant versions are

        H5PYDataset     0.1
        fuel.converters 0.1


Working with external packages
------------------------------

By default, Fuel looks for downloaders and converters in the
``fuel.downloaders`` and ``fuel.converters`` modules, respectively, but you're
not limited to that.

Fuel can be told to look into additional modules by setting the
``extra_downloaders`` and ``extra_converters`` configuration variables in
``~/.fuelrc``. These variables are expected to be lists of module names.

For instance, suppose you'd like to include the following modules:

* ``package1.extra_downloaders``
* ``package2.extra_downloaders``
* ``package1.extra_converters``
* ``package2.extra_converters``

You should include the following in your ``~/.fuelrc``:

.. code-block:: yaml

    # ~/.fuelrc
    extra_downloaders:
    - package1.extra_downloaders
    - package2.extra_downloaders
    extra_converters:
    - package1.extra_converters
    - package2.extra_converters

These configuration variables can be overridden through the
``FUEL_EXTRA_DOWNLOADERS`` and ``FUEL_EXTRA_CONVERTERS`` environment variables,
which are expected to be strings of space-separated module names, like so:

.. code-block:: bash

    export FUEL_EXTRA_DOWNLOADERS="package1.extra_downloaders package2.extra_downloaders"
    export FUEL_EXTRA_CONVERTERS="package1.extra_converters package2.extra_converters"

This feature lets external developers define their own Fuel dataset
downloader/converter packages, and also makes working with private datasets more
straightforward.
