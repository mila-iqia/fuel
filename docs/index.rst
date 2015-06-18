Welcome to Fuel's documentation!
================================

.. toctree::
   :hidden:

   overview
   built_in_datasets
   h5py_dataset
   new_dataset
   api/index

Fuel is a data pipeline framework which provides your machine learning models
with the data they need. It is planned to be used by both the Blocks_ and
Pylearn2_ neural network libraries.

* Fuel allows you to easily read different types of data (NumPy binary files,
  CSV files, HDF5 files, text files) using a single interface which is based on
  Python's iterator types.
* Provides a a series of wrappers around frequently used datasets such as
  MNIST, CIFAR-10 (vision), the One Billion Word Dataset (text corpus), and
  many more.
* Allows you iterate over data in a variety of ways, e.g. in order, shuffled,
  sampled, etc.
* Gives you the possibility to process your data on-the-fly through a series of
  (chained) transformation procedures. This way you can whiten your data,
  noise, rotate, crop, pad, sort or shuffle, cache it, and much more.
* Is pickle-friendly, allowing you to stop and resume long-running experiments
  in the middle of a pass over your dataset without losing any training
  progress.

.. warning::
   Fuel is a new project which is still under development. As such, certain
   (all) parts of the framework are subject to change. The last stable (but
   possibly outdated) release can be found in the ``stable`` branch.

.. tip::

   That said, if you are interested in using Fuel and run into any problems,
   feel free to ask your question on the `mailing list`_. Also, don't hesitate
   to file bug reports and feature requests by `making a GitHub issue`_.

.. _mailing list: https://groups.google.com/d/forum/fuel-users
.. _making a GitHub issue: https://github.com/mila-udem/fuel/issues/new
.. _Blocks: https://github.com/mila-udem/blocks
.. _Pylearn2: https://github.com/lisa-lab/pylearb2

Motivation
----------

Fuel was originally factored out of the Blocks_ framework in the hope of being
useful to other frameworks such as Pylearn2_ as well. It shares similarities
with the skdata_ package, but with a much heavier focus on data iteration and
processing.

.. _skdata: https://github.com/jaberg/skdata

Quickstart
----------

Begin by telling Fuel where to find the data it needs. You can do this by
creating a ``.fuelrc`` file:

.. code-block:: bash

   echo "data_path: /home/your_data" >> ~/.fuelrc

or by setting the environment variable ``FUEL_DATA_PATH``

.. code-block:: bash

   export FUEL_DATA_PATH=/home/your_data

This data path is a sequence of paths separated by an os-specific delimiter
(':' for Linux and OSX, ';' for Windows).

For example, after downloading the MNIST data to ``/home/your_data/mnist`` we
construct a handle to the data.

>>> from fuel.datasets import MNIST
>>> mnist = MNIST(which_sets=('train',))

In order to start reading the data, we need to initialize a *data stream*. A
data stream combines a dataset with a particular iteration scheme to read data
in a particular way. Let's say that in this case we want retrieve random
minibatches of size 512.

>>> from fuel.streams import DataStream
>>> from fuel.transformers import Flatten
>>> from fuel.schemes import ShuffledScheme
>>> stream = Flatten(
...     DataStream.default_stream(
...         mnist, iteration_scheme=ShuffledScheme(mnist.num_examples, 512)),
...     which_sources=('features',))

Datasets can apply various default transformations on the original
data stream if their ``apply_default_transformers`` method is called. A
convenient way to do so is to instantiate the data stream through the
``default_stream`` class method. In this case, MNIST rescaled pixel values in
the unit interval and flattened the images into vectors.

This stream can now provide us with a Python iterator which will provide a
total of 60,000 examples (``mnist.num_examples``) in the form of batches of
size 512. We call a single pass over the data an *epoch*, and hence the
iterator is called an *epoch iterator*.

>>> epoch = stream.get_epoch_iterator()

This iterator behaves like any other Python iterator, so we call :func:`next` on it

>>> next(epoch)  # doctest: +ELLIPSIS
(array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],...

and we can use a ``for`` loop

>>> for batch in epoch:
...     pass

Once we have completed the epoch, the iterator will be exhausted

>>> next(epoch)
Traceback (most recent call last):
  ...
StopIteration

but we can ask the stream for a new one, which will provide a complete
different set of minibatches.

>>> new_epoch = stream.get_epoch_iterator()

We can iterate over epochs as well, providing our model with an endless stream
of MNIST batches.

>>> for epoch in stream.iterate_epochs():  # doctest: +SKIP
...     for batch in epoch:
...         pass

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
