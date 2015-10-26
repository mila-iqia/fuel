Welcome to Fuel's documentation!
================================

.. toctree::
   :hidden:

   setup
   overview
   built_in_datasets
   h5py_dataset
   new_dataset
   extending_fuel
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
.. _Pylearn2: https://github.com/lisa-lab/pylearn2

Motivation
----------

Fuel was originally factored out of the Blocks_ framework in the hope of being
useful to other frameworks such as Pylearn2_ as well. It shares similarities
with the skdata_ package, but with a much heavier focus on data iteration and
processing.

.. _skdata: https://github.com/jaberg/skdata

Quickstart
==========

The best way to get started with Fuel is to have a look at the
:doc:`overview <overview>` documentation section.

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
