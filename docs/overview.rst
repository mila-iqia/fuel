Overview
========

We'll go over a quick example to see what Fuel is capable of.

Let's start by creating some random data to act as features and targets. We'll
pretend that we have eight 2x2 grayscale images separated into four classes.

>>> import numpy
>>> seed = 1234
>>> rng = numpy.random.RandomState(seed)
>>> features = rng.randint(256, size=(8, 2, 2))
>>> targets = rng.randint(4, size=(8, 1))

Our goal is to use Fuel to interface with this data, iterate over it in various
ways and apply transformations to it on the fly.

Division of labour
------------------

There are four basic tasks that Fuel needs to handle:

* Interface with the data, be it on disk or in memory.
* Decide which data points to visit, and in which order.
* Iterate over the selected data points.
* At each iteration step, apply some transformation to the selected data points.

Each of those four tasks is delegated to a particular class of objects, which
we'll be introducing in order.

Datasets: interfacing with data
-------------------------------

The :class:`Dataset` class is responsible for interfacing with the data and
handling data access requests. Subclasses of :class:`Dataset` specialize in
certain types of data.

IterableDataset
^^^^^^^^^^^^^^^

The simplest :class:`Dataset` subclass is :class:`IterableDataset`, which
interfaces with iterable objects.

It is created by passing a :class:`dict` mapping source names to their
associated data and, optionally, a :class:`dict` mapping source names to tuples
of axis labels.

>>> from collections import OrderedDict
>>> from fuel.datasets import IterableDataset
>>> dataset = IterableDataset(
...     iterables=OrderedDict([('features', features), ('targets', targets)]),
...     axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
...                              ('targets', ('batch', 'index'))]))

We can ask the dataset what sources of data it provides by accessing its
``sources`` attribute. We can also know which axes correspond to what by
accessing its ``axis_labels`` attribute. It also has a ``num_examples`` property
telling us the number of examples it contains.

>>> print('Sources are {}.'.format(dataset.sources))
Sources are ('features', 'targets').
>>> print('Axis labels are {}.'.format(dataset.axis_labels))
Axis labels are OrderedDict([('features', ('batch', 'height', 'width')), ('targets', ('batch', 'index'))]).
>>> print('Dataset contains {} examples.'.format(dataset.num_examples))
Dataset contains 8 examples.

.. tip::

   The source order of an :class:`IterableDataset` instance depends on the key
   order of ``iterables``, which is nondeterministic for regular :class:`dict`
   instances. We therefore recommend that you use
   :class:`collections.OrderedDict` instances if the source order is important
   to you.

Datasets themselves are stateless objects (as opposed to, say, an open file
handle, or an iterator object). In order to request data from the dataset, we
need to ask it to instantiate some stateful object with which it will interact.
This is done through the :meth:`Dataset.open` method:

>>> state = dataset.open()
>>> print(state.__class__.__name__)
imap

We see that in :class:`IterableDataset`'s case the state is an iterator object.
We can now visit the examples this dataset contains using its
:meth:`get_data` method.

>>> print(dataset.get_data(state=state))
(array([[ 47, 211],
       [ 38,  53]]), array([0]))
>>> while True:
...     try:
...         __ = dataset.get_data(state=state)
...     except StopIteration:
...         print('Iteration over')
...         break
Iteration over

Eventually, the iterator is depleted and it raises a :class:`StopIteration`
exception. We can iterate over the dataset again by requesting a fresh iterator
through the dataset's :meth:`reset` method.

>>> state = dataset.reset(state=state)
>>> print(dataset.get_data(state=state))
(array([[ 47, 211],
       [ 38,  53]]), array([0]))

When you're done, don't forget to call the dataset's :meth:`close` method on
the state. This has the effect of cleanly closing the state (e.g. if the state
is an open file handle, :meth:`close` will close it).

>>> dataset.close(state=state)

IndexableDataset
^^^^^^^^^^^^^^^^

The :class:`IterableDataset` implementation is pretty minimal. For instance, it
only lets you iterate sequentially and examplewise over your data.

If your data happens to be indexable (e.g. a :class:`list`, or a
:class:`numpy.ndarray`), then :class:`IndexableDataset` will let you do much
more.

We instantiate :class:`IndexableDataset` just like :class:`IterableDataset`.

>>> from fuel.datasets import IndexableDataset
>>> dataset = IndexableDataset(
...     indexables=OrderedDict([('features', features), ('targets', targets)]),
...     axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
...                              ('targets', ('batch', 'index'))]))

The main advantage of :class:`IndexableDataset` over :class:`IterableDataset`
is that it allows random access of the data it contains. In order to do so, we
need to pass an additional ``request`` argument to :meth:`get_data` in the form
of a list of indices.

>>> state = dataset.open()
>>> print('State is {}.'.format(state))
State is None.
>>> print(dataset.get_data(state=state, request=[0, 1]))
(array([[[ 47, 211],
        [ 38,  53]],
<BLANKLINE>
       [[204, 116],
        [152, 249]]]), array([[0],
       [3]]))
>>> dataset.close(state=state)

See how :class:`IndexableDataset` returns a ``None`` state: this is because
there's no actual state to maintain in this case.

Restricting sources
^^^^^^^^^^^^^^^^^^^

In some cases (e.g. unsupervised learning), you might want to use a subset of
the provided sources. This is achieved by passing a ``sources`` argument to the
dataset constructor. Here's an example:

>>> restricted_dataset = IndexableDataset(
...     indexables=OrderedDict([('features', features), ('targets', targets)]),
...     axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
...                              ('targets', ('batch', 'index'))]),
...     sources=('features',))
>>> state = restricted_dataset.open()
>>> print(restricted_dataset.get_data(state=state, request=[0, 1]))
(array([[[ 47, 211],
        [ 38,  53]],
<BLANKLINE>
       [[204, 116],
        [152, 249]]]),)
>>> restricted_dataset.close(state=state)

You can see that in this case only the features are returned by
:meth:`get_data`.

Iteration schemes: which examples to visit
------------------------------------------

Encapsulating and accessing our data is good, but if we're to integrate it into
a training loop, we need to be able to iterate over the data. For that, we need
to decide *which* indices to request and in *which order*. This is accomplished
via an :class:`IterationScheme` subclass.

At its most basic level, an iteration scheme is responsible, through its
:meth:`get_request_iterator` method, for building an iterator that will return
requests. Here are some examples:

>>> from fuel.schemes import (SequentialScheme, ShuffledScheme,
...                           SequentialExampleScheme, ShuffledExampleScheme)
>>> schemes = [SequentialScheme(examples=8, batch_size=4),
...            ShuffledScheme(examples=8, batch_size=4),
...            SequentialExampleScheme(examples=8),
...            ShuffledExampleScheme(examples=8)]
>>> for scheme in schemes:
...     print(list(scheme.get_request_iterator()))
[[0, 1, 2, 3], [4, 5, 6, 7]]
[[7, 2, 1, 6], [0, 4, 3, 5]]
[0, 1, 2, 3, 4, 5, 6, 7]
[7, 2, 1, 6, 0, 4, 3, 5]

We can therefore use an iteration scheme to visit a dataset in some order.

>>> state = dataset.open()
>>> scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=4)
>>> for request in scheme.get_request_iterator():
...     data = dataset.get_data(state=state, request=request)
...     print(data[0].shape, data[1].shape)
(4, 2, 2) (4, 1)
(4, 2, 2) (4, 1)
>>> dataset.close(state)

Data streams: automating the iteration process
----------------------------------------------

Iteration schemes offer a more convenient way to visit the dataset than
accessing the data by hand, but we can do better: the act of getting a fresh
state from the dataset, getting a request iterator from the iteration scheme,
using both to access the data and closing the state is repetitive. To automate
this, we have *data streams*, which are subclasses of
:class:`AbstractDataStream`.

The most common :class:`AbstractDataStream` subclass is :class:`DataStream`. It
is instantiated with a dataset and an iteration scheme, and returns an epoch
iterator through its :meth:`get_epoch_iterator` method, which iterates over the
dataset in the order defined by the iteration scheme.

>>> from fuel.streams import DataStream
>>> data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)
>>> for data in data_stream.get_epoch_iterator():
...     print(data[0].shape, data[1].shape)
(4, 2, 2) (4, 1)
(4, 2, 2) (4, 1)


Transformers: apply some transformation on the fly
--------------------------------------------------

Some data streams take data streams as input. We call them *transformers*, and
they enable us to build complex data preprocessing pipelines.

Transformers are :class:`Transformer` subclasses. Most of the the transformers
you'll encounter are located in the ``fuel.transformers`` module. Here are some
commonly used ones:

* :class:`Flatten`: flattens the input into a matrix (for batch input) or a
  vector (for examplewise input).
* :class:`ScaleAndShift`: scales and shifts the input by scalar quantities.
* :class:`Cast`: casts the input into some data type.

As an example, let's standardize the images we have by substracting their mean
and dividing by their standard deviation.

>>> from fuel.transformers import ScaleAndShift
>>> # Note: ScaleAndShift applies (batch * scale) + shift, as
>>> # opposed to (batch + shift) * scale.
>>> scale = 1.0 / features.std()
>>> shift = - scale * features.mean()
>>> standardized_stream = ScaleAndShift(data_stream=data_stream,
...                                     scale=scale, shift=shift,
...                                     which_sources=('features',))

The resulting data stream can be used to iterate over the dataset just like
before, but this time features will be standardized on-the-fly.

>>> for batch in standardized_stream.get_epoch_iterator():
...     print(batch)
(array([[[ 0.18530572, -1.54479571],
        [ 0.42249705,  0.24111545]],
<BLANKLINE>
       [[-1.30760439,  0.98059429],
        [-1.43317627, -1.2238898 ]],
<BLANKLINE>
       [[ 1.46892937,  1.58054882],
        [ 0.47830677, -1.2657471 ]],
<BLANKLINE>
       [[ 0.63178351, -0.28907693],
        [-0.40069638,  1.10616617]]]), array([[1],
       [0],
       [3],
       [2]]))
(array([[[ 1.32940506, -0.2332672 ],
        [-1.60060544, -0.31698179]],
<BLANKLINE>
       [[ 0.03182898,  0.50621164],
        [-1.64246273,  1.28754777]],
<BLANKLINE>
       [[ 0.88292727, -0.34488665],
        [ 0.15740086,  1.51078666]],
<BLANKLINE>
       [[-1.00065091, -0.84717417],
        [ 0.84106998, -0.19140991]]]), array([[2],
       [0],
       [3],
       [2]]))

Now, let's imagine that for some reason (e.g. running Theano code on GPU) we
need features to have a data type of ``float32``.

>>> from fuel.transformers import Cast
>>> cast_standardized_stream = Cast(
...     data_stream=standardized_stream,
...     dtype='float32', which_sources=('features',))

As you can see, Fuel makes it easy to chain transformations to form a
preprocessing pipeline. The complete pipeline now looks like this:

>>> data_stream = Cast(
...     ScaleAndShift(
...         DataStream(
...             dataset=dataset, iteration_scheme=scheme),
...         scale=scale, shift=shift, which_sources=('features',)),
...     dtype='float32', which_sources=('features',))

Schematic overview of Fuel
--------------------------

For the more visual people, here's a schematic view of how the different
components of Fuel interact together.

.. digraph:: datasets
   :caption: A simplified overview of the interactions between the different parts of the data-handling classes in Fuel. Dashed lines are optional.

   Dataset -> DataStream [label=" Argument to"];
   DataStream -> Dataset [label=" Gets data from"];
   DataStream -> DataIterator [label=" Returns"];
   IterationScheme -> DataStream [style=dashed, label=" Argument to"];
   DataStream -> IterationScheme [style=dashed, label=" Gets request iterator"];
   IterationScheme -> RequestIterator [label=" Returns"];
   RequestIterator -> DataIterator [style=dashed, label=" Argument to"];
   DataIterator -> DataStream [label=" Gets data from"];
   DataStream -> DataStream [style=dashed, label=" Gets data from (transformer)"];
   { rank=same; RequestIterator DataIterator }

Datasets
  Datasets provide an interface to the data we are trying to access. This data
  is usually stored on disk, but can also be created on the fly (e.g. drawn
  from a distribution), requested from a database or server, etc. Datasets are
  largely *stateless*. Multiple data streams can be iterating over the same
  dataset simultaneously, so the dataset couldn't have a single state to store
  e.g. its location in a file. Instead, the dataset provides a set of methods
  (:meth:`~.datasets.Dataset.open`, :meth:`~.datasets.Dataset.close`,
  :meth:`~.datasets.Dataset.get_data`, etc.) that interact with a particular
  state, which is managed by a data stream.

Data stream
  A data stream uses the interface of a dataset to e.g. iterate over the data.
  Data streams can produce data set iterators (epoch iterators) which will use
  the data stream's state to return actual data. Data streams can optionally
  use an iteration scheme to describe in what way (e.g. in what order) they
  will request data from the dataset.

Transformer
  A transformer is really just another data stream, except that it doesn't take
  a dataset but another data stream as its input. This allows us to set up a
  data processing pipeline, which can be quite powerful. For example, given a
  data set that produces sentences from a text corpus, we could use a chain of
  transformers to read groups of sentences into a cache, sort them by length,
  group them into minibatches, and pad them to be of the same length.

Iteration scheme
  A iteration scheme describes *how* we should proceed to iterate over the
  data. Iteration schemes will normally describe a sequence of batch sizes
  (e.g.  a constant minibatch size), or a sequence of indices to our data (e.g.
  indices of shuffled minibatches). Iteration schemes return request iterators.

Request iterator
  A request iterator implements the Python iteration protocol. It represents a
  single epoch of requests, as determined by the iteration scheme that produced
  it.

Data iterator
  A data iterator also implements the Python iteration protocol. It optionally
  uses a request iterator and returns data at each step (requesting it from the
  data stream). A single iteration over a data iterator represents a single
  epoch.

Going further
-------------

You now know enough to find your way around Fuel. Here are the next steps:

* Learn :doc:`how to use built-in datasets <built_in_datasets>`.
* Learn :doc:`how to import your own data in Fuel <h5py_dataset>`.
* Learn :doc:`how to extend Fuel <extending_fuel>` to suit your needs.
