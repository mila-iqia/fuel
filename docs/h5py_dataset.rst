Getting your data in Fuel
=========================

.. warning::

    We're still in the process of figuring out the interface, which means
    the "preferred" way of getting your data in Fuel may change in the future.

Built-in datasets are convenient for training on common benchmark tasks, but
what if you want to train a model on your own data?

This section shows how to accomplish the common task of loading into Fuel a
bunch of data sources (*e.g.* features, targets) split into different sets.

A toy example
-------------

We'll start this tutorial by creating bogus data sources that could be lying
around on disk.

>>> import numpy
>>> numpy.save(
...     'train_vector_features.npy',
...     numpy.random.normal(size=(90, 10)).astype('float32'))
>>> numpy.save(
...     'test_vector_features.npy',
...     numpy.random.normal(size=(10, 10)).astype('float32'))
>>> numpy.save(
...     'train_image_features.npy',
...     numpy.random.randint(2, size=(90, 3, 5, 5)).astype('uint8'))
>>> numpy.save(
...     'test_image_features.npy',
...     numpy.random.randint(2, size=(10, 3, 5, 5)).astype('uint8'))
>>> numpy.save(
...     'train_targets.npy',
...     numpy.random.randint(10, size=(90, 1)).astype('uint8'))
>>> numpy.save(
...     'test_targets.npy',
...     numpy.random.randint(10, size=(10, 1)).astype('uint8'))

Our goal is to process these files into a format that can be natively imported
in Fuel.

HDF5 datasets
-------------

The best-supported way to load data in Fuel is through the
:class:`~.datasets.hdf5.H5PYDataset` class, which wraps HDF5 files using
``h5py``.

This is the class that's used for most built-in datasets. It makes a series of
assumptions about the structure of the HDF5 file which greatly simplify
things if your data happens to meet these assumptions:

* All data is stored into a single HDF5 file.
* Data sources reside in the root group, and their names define the source
  names.
* Data sources are not explicitly split. Instead, splits are defined as
  attributes of the root group. They're expected to be numpy arrays of
  shape ``(2,)``, with the first element being the starting point
  (inclusive) of the split and the last element being the stopping
  point (exclusive) of the split.

.. tip::

    Some of you may wonder if this means all data has to be read off disk all
    the time. Rest assured, :class:`~.datasets.hdf5.H5PYDataset` has an
    option to load things into memory which we will be covering soon.

Converting the toy example
--------------------------

Let's now convert our bogus files into a format that's digestible by
:class:`~.datasets.hdf5.H5PYDataset`.

We first load the data from disk.

>>> train_vector_features = numpy.load('train_vector_features.npy')
>>> test_vector_features = numpy.load('test_vector_features.npy')
>>> train_image_features = numpy.load('train_image_features.npy')
>>> test_image_features = numpy.load('test_image_features.npy')
>>> train_targets = numpy.load('train_targets.npy')
>>> test_targets = numpy.load('test_targets.npy')

We then open an HDF5 file for writing and create three datasets in the root
group, one for each data source. We name them after their source name.

>>> import h5py
>>> f = h5py.File('dataset.hdf5', mode='w')
>>> vector_features = f.create_dataset(
...     'vector_features', (100, 10), dtype='float32')
>>> image_features = f.create_dataset(
...     'image_features', (100, 3, 5, 5), dtype='uint8')
>>> targets = f.create_dataset(
...     'targets', (100, 1), dtype='uint8')

Notice how the number of examples we specify (100) in the shapes is the sum of
the number of training and test examples. We'll be filling the first 90 rows
with training examples and the last 10 rows with test examples.

>>> vector_features[...] = numpy.vstack(
...     [train_vector_features, test_vector_features])
>>> image_features[...] = numpy.vstack(
...     [train_image_features, test_image_features])
>>> targets[...] = numpy.vstack([train_targets, test_targets])

The last thing we need to do is to give :class:`~.datasets.hdf5.H5PYDataset`
a way to recover what the splits are. This is done by setting attributes in
the root group.

>>> f.attrs['train'] = [0, 90]
>>> f.attrs['test'] = [90, 100]

We flush, close the file and *voilà*!

>>> f.flush()
>>> f.close()

Playing with H5PYDataset datasets
---------------------------------

Let's explore what we can do with the dataset we just created.

The simplest thing is to load it by giving its path:

>>> from fuel.datasets.hdf5 import H5PYDataset
>>> dataset = H5PYDataset('dataset.hdf5')

By default, the whole data is used:

>>> print(dataset.num_examples)
100

In order to use the training set or the test set, you need to pass a
``which_set`` argument:

>>> train_set = H5PYDataset('dataset.hdf5', which_set='train')
>>> print(train_set.num_examples)
90
>>> test_set = H5PYDataset('dataset.hdf5', which_set='test')
>>> print(test_set.num_examples)
10

You can further restrict which examples are used by providing a ``slice`` object
as the ``subset`` argument. *Make sure that its* ``step`` *is either 1 or*
``None`` *, as these are the only two options that are supported*.

>>> train_set = H5PYDataset(
...     'dataset.hdf5', which_set='train', subset=slice(0, 80))
>>> print(train_set.num_examples)
80
>>> valid_set = H5PYDataset(
...     'dataset.hdf5', which_set='train', subset=slice(80, 90))
>>> print(valid_set.num_examples)
10

The available data sources are defined by the names of the datasets in the root
node of the HDF5 file, and :class:`~.datasets.hdf5.H5PYDataset` automatically
picked them up for us:

>>> print(train_set.provides_sources) # doctest: +SKIP
[u'image_features', u'targets', u'vector_features']

We can request data as usual:

>>> handle = train_set.open()
>>> data = train_set.get_data(handle, slice(0, 10))
>>> print((data[0].shape, data[1].shape, data[2].shape))
((10, 3, 5, 5), (10, 1), (10, 10))
>>> train_set.close(handle)

We can also request just the vector features:

>>> train_vector_features = H5PYDataset(
...     'dataset.hdf5', which_set='train', subset=slice(0, 80),
...     sources=['vector_features'])
>>> handle = train_vector_features.open()
>>> data, = train_vector_features.get_data(handle, slice(0, 10))
>>> print(data.shape)
(10, 10)
>>> train_vector_features.close(handle)

Loading data in memory
----------------------

Reading data off disk is inefficient compared to storing it in memory. Large
datasets make it inevitable, but if your dataset is small enough that it fits
into memory, you should take advantage of it.

In :class:`~.datasets.hdf5.H5PYDataset`, this is accomplished via the
``load_in_memory`` constructor argument. It has the effect of loading *just*
what you requested, and nothing more.

>>> in_memory_train_vector_features = H5PYDataset(
...     'dataset.hdf5', which_set='train', subset=slice(0, 80),
...     sources=['vector_features'], load_in_memory=True)
>>> data, = in_memory_train_vector_features.data_sources
>>> print(type(data)) # doctest: +SKIP
<type 'numpy.ndarray'>
>>> print(data.shape)
(80, 10)
