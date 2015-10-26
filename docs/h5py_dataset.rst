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
* Data sources are not explicitly split into separate HDF5 datasets or separate
  HDF5 files. Instead, splits are defined in the ``split`` attribute of the root
  group. It's expected to be a 1D numpy array of compound ``dtype`` with seven
  fields, organized as follows:

  1. ``split`` : string identifier for the split name
  2. ``source`` : string identifier for the source name
  3. ``start`` : start index (inclusive) of the split in the source
     array, used if ``indices`` is a null reference.
  4. ``stop`` : stop index (exclusive) of the split in the source
     array, used if ``indices`` is a null reference.
  5. ``indices`` : h5py.Reference, reference to a dataset containing
     subset indices for this split/source pair. If it's a null
     reference, ``start`` and ``stop`` are used.
  6. ``available`` : boolean, ``False`` is this split is not available
     for this source
  7. ``comment`` : comment string

.. tip::

    Some of you may wonder if this means all data has to be read off disk all
    the time. Rest assured, :class:`~.datasets.hdf5.H5PYDataset` has an
    option to load things into memory which we will be covering soon.

.. _convert_h5py_dataset:

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

:class:`~.datasets.hdf5.H5PYDataset` allows us to label axes with semantic
information. We record that information in the HDF5 file through `dimension
scales`_.

>>> vector_features.dims[0].label = 'batch'
>>> vector_features.dims[1].label = 'feature'
>>> image_features.dims[0].label = 'batch'
>>> image_features.dims[1].label = 'channel'
>>> image_features.dims[2].label = 'height'
>>> image_features.dims[3].label = 'width'
>>> targets.dims[0].label = 'batch'
>>> targets.dims[1].label = 'index'

This particular choice of label is arbitrary. Nothing in Fuel forces you to
adopt any labeling convention. Note, however, that certain external frameworks
that rely on Fuel *may* impose some restrictions on the choice of labels.

The last thing we need to do is to give :class:`~.datasets.hdf5.H5PYDataset`
a way to recover what the splits are. This is done by setting the ``split``
attribute of the root group.

>>> split_array = numpy.empty(
...     6,
...     dtype=numpy.dtype([
...         ('split', 'a', 5),
...         ('source', 'a', 15),
...         ('start', numpy.int64, 1),
...         ('stop', numpy.int64, 1),
...         ('indices', h5py.special_dtype(ref=h5py.Reference)),
...         ('available', numpy.bool, 1),
...         ('comment', 'a', 1)]))
>>> split_array[0:3]['split'] = 'train'.encode('utf8')
>>> split_array[3:6]['split'] = 'test'.encode('utf8')
>>> split_array[0:6:3]['source'] = 'vector_features'.encode('utf8')
>>> split_array[1:6:3]['source'] = 'image_features'.encode('utf8')
>>> split_array[2:6:3]['source'] = 'targets'.encode('utf8')
>>> split_array[0:3]['start'] = 0
>>> split_array[0:3]['stop'] = 90
>>> split_array[3:6]['start'] = 90
>>> split_array[3:6]['stop'] = 100
>>> split_array[:]['indices'] = h5py.Reference()
>>> split_array[:]['available'] = True
>>> split_array[:]['comment'] = '.'.encode('utf8')
>>> f.attrs['split'] = split_array

We created a 1D numpy array with six elements. The ``dtype`` for this array
is a compound type: every element of the array is a tuple of ``(str, str, int,
int, h5py.Reference, bool, str)``. The length of each string element has been
chosen to be the maximum length we needed to store: that's 5 for the ``split``
element (``'train'`` being the longest split name) and 15 for the ``source``
element (``'vector_features'`` being the longest source name). We didn't
include any comment, so the length for that element was set to 1. Due to a
quirk in pickling empty strings, we put ``'.'`` as the comment value.

.. warning::

    Due to limitations in h5py, you must make sure to use bytes for ``split``,
    ``source`` and ``comment``.

:class:`~.datasets.hdf5.H5PYDataset` expects the ``split`` attribute of the
root node to contain as many elements as the cartesian product of all sources
and all splits, *i.e.* all possible split/source combinations. Sometimes, no
data is available for some source/split combination: for instance, the test
set may not be labeled, and the ``('test', 'targets')`` combination may not
exist. In that case, you can set the ``available`` element for that combination
to ``False``, and :class:`~.datasets.hdf5.H5PYDataset` will ignore it.

Don't worry too much about ``indices``; we'll get back to that later. For the
moment, all you need to know is that since our splits are contiguous, we don't
need that feature and therefore put empty references.

The method described above does the job, but it's not very convenient. An even
simpler way of achieving the same result is to call
:meth:`~.datasets.hdf5.H5PYDataset.create_split_array`.

>>> from fuel.datasets.hdf5 import H5PYDataset
>>> split_dict = {
...     'train': {'vector_features': (0, 90), 'image_features': (0, 90),
...               'targets': (0, 90)},
...     'test': {'vector_features': (90, 100), 'image_features': (90, 100),
...              'targets': (90, 100)}}
>>> f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

The :meth:`~.datasets.hdf5.H5PYDataset.create_split_array` method expects
a dictionary mapping split names to dictionaries. Those dictionaries map source
names to tuples of length 2, 3 or 4. The first two elements correspond to the
start and stop indexes. The other two elements are optional and correspond to
the ``indices`` reference and the comment, respectively. The method will create
the array behind the scenes, choose the string lengths automatically and
populate it with the information in the split dictionary. If a particular
split/source combination isn't present, its ``available`` attribute is set to
``False``, which allows us to specify only what's actually present in the HDF5
file we created.

.. tip::

    By default, :class:`~.datasets.hdf5.H5PYDataset` sorts sources in
    alphabetical order, and data requests are also returned in that order. If
    ``sources`` is passed as argument upon instantiation,
    :class:`~.datasets.hdf5.H5PYDataset` will use the order of ``sources``
    instead. This means that if you want to force a particular source order, you
    can do so by explicitly passing the ``sources`` argument with the desired
    ordering. For example, if your dataset has two sources named ``'features'``
    and ``'targets'`` and you'd like the targets to be returned first, you need
    to pass ``sources=('targets', 'features')`` as a constructor argument.

We flush, close the file and *voilà*!

>>> f.flush()
>>> f.close()

Playing with H5PYDataset datasets
---------------------------------

Let's explore what we can do with the dataset we just created.

The simplest thing is to load it by giving its path and a tuple of split names:

>>> train_set = H5PYDataset('dataset.hdf5', which_sets=('train',))
>>> print(train_set.num_examples)
90
>>> test_set = H5PYDataset('dataset.hdf5', which_sets=('test',))
>>> print(test_set.num_examples)
10

Passing more than one split name would cause the splits to be concatenated.
The available data sources would be the intersection of the sources provided
by each split.

You can further restrict which examples are used by providing a ``slice`` object
or a list of indices as the ``subset`` argument.

>>> train_set = H5PYDataset(
...     'dataset.hdf5', which_sets=('train',), subset=slice(0, 80))
>>> print(train_set.num_examples)
80
>>> valid_set = H5PYDataset(
...     'dataset.hdf5', which_sets=('train',), subset=slice(80, 90))
>>> print(valid_set.num_examples)
10

The available data sources are defined by the names of the datasets in the root
node of the HDF5 file, and :class:`~.datasets.hdf5.H5PYDataset` automatically
picked them up for us:

>>> print(train_set.provides_sources)
('image_features', 'targets', 'vector_features')

It also parsed axis labels, which are accessible through the ``axis_labels``
property, which is a dict mapping source names to a tuple of axis labels:

>>> print(train_set.axis_labels['image_features'])
('batch', 'channel', 'height', 'width')
>>> print(train_set.axis_labels['vector_features'])
('batch', 'feature')
>>> print(train_set.axis_labels['targets'])
('batch', 'index')

We can request data as usual:

>>> handle = train_set.open()
>>> data = train_set.get_data(handle, slice(0, 10))
>>> print((data[0].shape, data[1].shape, data[2].shape))
((10, 3, 5, 5), (10, 1), (10, 10))
>>> train_set.close(handle)

We can also request just the vector features:

>>> train_vector_features = H5PYDataset(
...     'dataset.hdf5', which_sets=('train',), subset=slice(0, 80),
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
...     'dataset.hdf5', which_sets=('train',), subset=slice(0, 80),
...     sources=['vector_features'], load_in_memory=True)
>>> data, = in_memory_train_vector_features.data_sources
>>> print(type(data)) # doctest: +ELLIPSIS
<... 'numpy.ndarray'>
>>> print(data.shape)
(80, 10)

Non-contiguous splits
---------------------

Sometimes it's not possible to store the different splits contiguously. In that
case, you'll want to use the ``indices`` field of the
:class:`~.datasets.hdf5.H5PYDataset` split array. A non-empty reference in that
field overrides the ``start`` and ``stop`` fields, and the dataset the
reference points to is used to determine the indices for that split/source
pair.

Suppose that you'd like to use the even examples as your training set and the
odd examples as your test set. We'll start with the HDF5 file we populated
earlier and manipulate its ``split`` attribute.

>>> f = h5py.File('dataset.hdf5', mode='a')
>>> f['train_indices'] = numpy.arange(0, 100, 2)
>>> train_ref = f['train_indices'].ref
>>> f['test_indices'] = numpy.arange(1, 100, 2)
>>> test_ref = f['test_indices'].ref
>>> split_dict = {
...     'train': {'vector_features': (-1, -1, train_ref),
...               'image_features': (-1, -1, train_ref),
...               'targets': (-1, -1, train_ref)},
...     'test': {'vector_features': (-1, -1, test_ref),
...              'image_features': (-1, -1, test_ref),
...              'targets': (-1, -1, test_ref)}}
>>> f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
>>> f.flush()
>>> f.close()

We created two new datasets containing even and odd indices from 0 to 99,
respectively, and passed references to these datasets in the split dict. In that
case, the value we pass to ``start`` and ``stop`` really doesn't matter, so
we arbitrarily chose ``-1`` for both.

Let's check that the training and test set do contain even and odd examples:

>>> train_set = H5PYDataset(
...     'dataset.hdf5', which_sets=('train',), sources=('vector_features',))
>>> handle = train_set.open()
>>> print(
...     numpy.array_equal(
...         train_set.get_data(handle, slice(0, 50))[0],
...         numpy.vstack(
...             [numpy.load('train_vector_features.npy'),
...              numpy.load('test_vector_features.npy')])[::2]))
True
>>> train_set.close(handle)
>>> test_set = H5PYDataset(
...     'dataset.hdf5', which_sets=('test',), sources=('vector_features',))
>>> handle = test_set.open()
>>> print(
...     numpy.array_equal(
...         test_set.get_data(handle, slice(0, 50))[0],
...         numpy.vstack(
...             [numpy.load('train_vector_features.npy'),
...              numpy.load('test_vector_features.npy')])[1::2]))
True
>>> test_set.close(handle)

Variable-length data
--------------------

:class:`~.datasets.hdf5.H5PYDataset` also supports variable length data. Let's
update the image features to reflect that:

.. doctest::
   :hide:

   >>> numpy.random.seed(1234)

>>> sizes = numpy.random.randint(3, 9, size=(100,))
>>> train_image_features = [
...     numpy.random.randint(256, size=(3, size, size)).astype('uint8')
...     for size in sizes[:90]]
>>> test_image_features = [
...     numpy.random.randint(256, size=(3, size, size)).astype('uint8')
...     for size in sizes[90:]]

In this new example, images have random shapes ranging from 3x3 pixels to 8x8
pixels.

First, we put the vector features and the targets inside the HDF5 file as
before:

>>> f = h5py.File('dataset.hdf5', mode='w')
>>> f['vector_features'] = numpy.vstack(
...     [numpy.load('train_vector_features.npy'),
...      numpy.load('test_vector_features.npy')])
>>> f['targets'] = numpy.vstack(
...     [numpy.load('train_targets.npy'),
...      numpy.load('test_targets.npy')])
>>> f['vector_features'].dims[0].label = 'batch'
>>> f['vector_features'].dims[1].label = 'feature'
>>> f['targets'].dims[0].label = 'batch'
>>> f['targets'].dims[1].label = 'index'

We now have to put the variable-length images inside the HDF5 file. We can't
do that directly, since HDF5 and h5py don't support multi-dimensional ragged
arrays. However, there *is* support for 1D ragged arrays. Instead, we'll
flatten the images before putting them in the HDF5 file:

>>> all_image_features = train_image_features + test_image_features
>>> dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
>>> image_features = f.create_dataset('image_features', (100,), dtype=dtype)
>>> image_features[...] = [image.flatten() for image in all_image_features]
>>> image_features.dims[0].label = 'batch'

If you're feeling lost, have a look at the `dedicated tutorial on
variable-length data`_.

The images are now in the HDF5 file, but that doesn't help us unless we can
recover their original shape. For that, we'll create a dimension scale that
we'll attach to the ``'image_features'`` dataset using the name ``'shapes'``
(use this *exact* name):

>>> image_features_shapes = f.create_dataset(
...     'image_features_shapes', (100, 3), dtype='int32')
>>> image_features_shapes[...] = numpy.array(
...     [image.shape for image in all_image_features])
>>> image_features.dims.create_scale(image_features_shapes, 'shapes')
>>> image_features.dims[0].attach_scale(image_features_shapes)

We'd also like to tag those variable-length dimensions with semantic
information. We'll create another dimension scale that we'll attach to the
``'image_features'`` dataset using the name ``'shape_labels'``
(use this *exact* name):

>>> image_features_shape_labels = f.create_dataset(
...     'image_features_shape_labels', (3,), dtype='S7')
>>> image_features_shape_labels[...] = [
...     'channel'.encode('utf8'), 'height'.encode('utf8'),
...     'width'.encode('utf8')]
>>> image_features.dims.create_scale(
...     image_features_shape_labels, 'shape_labels')
>>> image_features.dims[0].attach_scale(image_features_shape_labels)

The :class:`~.datasets.hdf5.H5PYDataset` class will handle things from
there on. When image features are loaded, it will retrieve their shapes and
do the reshape automatically.

Lastly, we create the split dictionary exactly as before:

>>> split_dict = {
...     'train': {'vector_features': (0, 90), 'image_features': (0, 90),
...               'targets': (0, 90)},
...     'test': {'vector_features': (90, 100), 'image_features': (90, 100),
...              'targets': (90, 100)}}
>>> f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
>>> f.flush()
>>> f.close()

That's it. Now let's kick the tires a little. The axis labels appear as they
should:

>>> train_set = H5PYDataset(
...     'dataset.hdf5', which_sets=('train',), sources=('image_features',))
>>> print(train_set.axis_labels['image_features'])
('batch', 'channel', 'height', 'width')

:class:`~.datasets.hdf5.H5PYDataset` retrieves images of different shapes and
automatically unflattens them:

>>> handle = train_set.open()
>>> images, = train_set.get_data(handle, slice(0, 10))
>>> train_set.close(handle)
>>> print(images[0].shape, images[1].shape)
(3, 6, 6) (3, 8, 8)

The object returned by ``get_data`` is a 1D numpy array of objects:

>>> print(type(images), images.dtype, images.shape) # doctest: +ELLIPSIS
<... 'numpy.ndarray'> object (10,)

.. doctest::
   :hide:

   >>> import os
   >>> os.remove('train_image_features.npy')
   >>> os.remove('train_vector_features.npy')
   >>> os.remove('train_targets.npy')
   >>> os.remove('test_image_features.npy')
   >>> os.remove('test_vector_features.npy')
   >>> os.remove('test_targets.npy')
   >>> os.remove('dataset.hdf5')

.. _dimension scales: http://docs.h5py.org/en/latest/high/dims.html
.. _dedicated tutorial on variable-length data: http://docs.h5py.org/en/latest/special.html#arbitrary-vlen-data
