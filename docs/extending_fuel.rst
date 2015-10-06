Extending Fuel
==============

In this section, we'll cover how to extend three important components of Fuel:

* Dataset classes
* Iteration schemes
* Transformers

.. warning::
    The code presented in this section is for illustration purposes only and
    is not intended to be used outside of this tutorial.

New dataset classes
-------------------

.. admonition:: In summary
    :class: tip

    * Subclass from :class:`Dataset`.

      - Implement the :meth:`get_data` method.
      - If your dataset interacts with stateful objects (e.g. files on disk),
        override the :meth:`open` and :meth:`close` methods.

    * Subclass from :class:`IndexableDataset` if your data fits in memory.

      - :class:`IndexableDataset` constructor accepts an ``indexables``
        argument, which is expected to be a :class:`dict` mapping from source
        names to their corresponding data.

New dataset classes are implemented by subclassing :class:`Dataset` and
implementing its :meth:`get_data` method. If your dataset interacts with
stateful objects (e.g. files on disk), then you should also override the
:meth:`open` and :meth:`close` methods.

If your data fits in memory, you can save yourself some time by inheriting from
:class:`IndexableDataset`. In that case, all you need to do is load the data as
a :class:`dict` mapping source names to their corresponding data and pass it to
the superclass as the ``indexables`` argument.

For instance, here's how you would implement a specialized class to interface
with ``.npy`` files.

>>> from collections import OrderedDict
>>> import numpy
>>> from six import iteritems
>>> from fuel.datasets import IndexableDataset
>>>
>>> class NPYDataset(IndexableDataset):
...     def __init__(self, source_paths, **kwargs):
...         indexables = OrderedDict(
...             [(source, numpy.load(path)) for
...              source, path in iteritems(source_paths)])
...         super(NPYDataset, self).__init__(indexables, **kwargs)

Here's this class in action:

>>> numpy.save('npy_dataset_features.npy',
...            numpy.arange(40).reshape((10, 4)))
>>> numpy.save('npy_dataset_targets.npy',
...            numpy.arange(10).reshape((10, 1)))
>>> dataset = NPYDataset(OrderedDict([('features', 'npy_dataset_features.npy'),
...                                   ('targets', 'npy_dataset_targets.npy')]))
>>> state = dataset.open()
>>> print(dataset.get_data(state=state, request=[0, 1, 2, 3]))
(array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]]), array([[0],
       [1],
       [2],
       [3]]))
>>> dataset.close(state)

.. doctest::
   :hide:

   >>> import os
   >>> os.remove('npy_dataset_features.npy')
   >>> os.remove('npy_dataset_targets.npy')

New iteration schemes
---------------------

.. admonition:: In summary
    :class: tip

    * Subclass from :class:`IterationScheme`.

      - Implement the :meth:`get_request_iterator` method, which should return
        an iterator object.

    * Subclass from :class:`IndexScheme` for schemes requesting examples.

      - :class:`IndexScheme` constructor accepts an ``examples`` argument,
        which can either be an integer or a list of indices.

    * Subclass from :class:`BatchScheme` for schemes requesting batches.

      - :class:`BatchScheme` constructor accepts an ``examples`` argument,
        which can either be an integer or a list of indices, and a
        ``batch_size`` argument.

New iteration schemes are implemented by subclassing :class:`IterationScheme`
and implementing a :meth:`get_request_iterator` method, which should return an
iterator that returns lists of indices.

Two subclasses of :class:`IterationScheme` typically serve as a basis for other
iteration schemes: :class:`IndexScheme` (for schemes requesting examples) and
:class:`BatchScheme` (for schemes requesting batches). Both subclasses are
instantiated by providing a list of indices or the number of examples to visit
(meaning that examples 0 through ``examples - 1`` would be visited),
and :class:`BatchScheme` accepts an additional ``batch_size`` argument.

Here's how you would implement an iteration scheme that iterates over even
examples:

>>> from fuel.schemes import IndexScheme, BatchScheme
>>> # `iter_` : A picklable version of `iter`
>>> from picklable_itertools import iter_, imap
>>> # Partition all elements of a sequence into tuples of length at most n
>>> from picklable_itertools.extras import partition_all

>>> class ExampleEvenScheme(IndexScheme):
...     def get_request_iterator(self):
...         indices = list(self.indices)[::2]
...         return iter_(indices)
>>> class BatchEvenScheme(BatchScheme):
...     def get_request_iterator(self):
...         indices = list(self.indices)[::2]
...         return imap(list, partition_all(self.batch_size, indices))

.. note::

    The ``partition_all`` function splits a sequence into chunks of size
    ``n`` (``self.batch_size``, in our case), with the last batch possibly being
    shorter if the length of the sequence is not a multiple of ``n``. It
    produces an iterator which returns these batches as tuples.

    The ``imap`` function applies a function to all elements of a sequence. It
    produces an iterator which returns the result of the function being applied
    to elements of the sequence. In our case, this has the effect of casting
    tuples into lists.

    The reason why we go through all this trouble is to produce an iterator,
    which is what :meth:`get_request_iterator` is supposed to return.

Here are the two iteration scheme classes in action:

>>> print(list(ExampleEvenScheme(10).get_request_iterator()))
[0, 2, 4, 6, 8]
>>> print(list(BatchEvenScheme(10, 2).get_request_iterator()))
[[0, 2], [4, 6], [8]]

New transformers
----------------

.. admonition:: In summary
    :class: tip

    * Subclass from :class:`Transformer`.

      - Implement :meth:`transform_example` if your transformer works on single
        examples and/or :meth:`transform_batch` if it works on batches.

    * Subclass from :class:`AgnosticTransformer` if the example and batch
      implementations are the same.

      - Implement the :meth:`transform_any` method.

    * Subclass from :class:`SourcewiseTransformer` if your transformation is
      applied sourcewise.

      - :class:`SourcewiseTransformer` constructor takes an additional
        ``which_sources`` keyword argument.
      - If ``which_sources`` is ``None``, then the transformation is applied to
        all sources.
      - Implement the :meth:`transform_source_example` and/or
        :meth:`transform_source_batch` methods.

    * Subclass from :class:`AgnosticSourcewiseTransformer` if your
      transformation is applied sourcewise and its example and batch
      implementations are the same.

      - Implement the :meth:`transform_any_source` method.

    * Consider using the :class:`Mapping` transformer (the swiss-army knife of
      transformers) for one-off transformations instead of implementing a
      subclass.

      - Its constructor accepts a ``mapping`` argument, which is expected to be
        a function taking a tuple of sources as input and returning the
        transformed sources.

An important thing to know about data streams is that they distinguish
between two types of outputs: single examples, and batches of examples.
Depending on your choice of iteration scheme, a data stream's
``produces_examples`` property will either be ``True`` (it produces examples)
or ``False`` (it produces batches).

Transformers are aware of this, and as such implement two distinct methods:
:meth:`transform_example` and :meth:`transform_batch`. A new transformer is
typically implemented by subclassing Transformer and implementing one or both
of these methods.

As an example, here's how you would double the value of some ``'features'``
data source.

>>> from fuel.transformers import Transformer
>>>
>>> class FeaturesDoubler(Transformer):
...     def __init__(self, data_stream, **kwargs):
...         super(FeaturesDoubler, self).__init__(
...             data_stream=data_stream,
...             produces_examples=data_stream.produces_examples,
...             **kwargs)
...         
...     def transform_example(self, example):
...         if 'features' in self.sources:
...             example = list(example)
...             index = self.sources.index('features')
...             example[index] *= 2
...             example = tuple(example)
...         return example
...     
...     def transform_batch(self, batch):
...         if 'features' in self.sources:
...             batch = list(batch)
...             index = self.sources.index('features')
...             batch[index] *= 2
...             batch = tuple(batch)
...         return batch

Since we wish to support both batches and examples, we'll declare our output
type to be the same as our data stream's output type.

If you were to build a transformer that only works on batches, you would pass
``produces_examples=False`` and implement only :meth:`transform_batch`. If
anyone tried to use your transformer on an example data stream, an error would
automatically be raised.

.. note::

    When applicable, it is good practice that a new transformer's constructor
    calls its superclass constructor by passing the data stream it receives and
    declaring whether it produce examples or batches.

Let's test our doubler on some dummy dataset.

>>> from fuel.schemes import SequentialExampleScheme, SequentialScheme
>>> from fuel.streams import DataStream
>>>
>>> dataset = IndexableDataset(
...     indexables=OrderedDict([
...         ('features', numpy.array([1, 2, 3, 4])),
...         ('targets', numpy.array([-1, 1, -1, 1]))]))
>>> example_scheme = SequentialExampleScheme(examples=dataset.num_examples)
>>> example_stream = FeaturesDoubler(
...     data_stream=DataStream(
...         dataset=dataset, iteration_scheme=example_scheme))
>>> batch_scheme = SequentialScheme(
...     examples=dataset.num_examples, batch_size=2)
>>> batch_stream = FeaturesDoubler(
...     data_stream=DataStream(
...         dataset=dataset, iteration_scheme=batch_scheme))
>>> print([example for example in example_stream.get_epoch_iterator()])
[(2, -1), (4, 1), (6, -1), (8, 1)]
>>> print([batch for batch in batch_stream.get_epoch_iterator()])
[(array([2, 4]), array([-1,  1])), (array([6, 8]), array([-1,  1]))]

You may have noticed that in this example the :meth:`transform_example` and
:meth:`transform_batch` implementations are the same. In such cases, you can
subclass from :class:`AgnosticTransformer` instead. It requires that you
implement a :meth:`transform_any` method, which will be called by both
:meth:`transform_example` and :meth:`transform_batch`.

>>> from fuel.transformers import AgnosticTransformer
>>> 
>>> class FeaturesDoubler(AgnosticTransformer):
...     def __init__(self, data_stream, **kwargs):
...         super(FeaturesDoubler, self).__init__(
...             data_stream=data_stream,
...             produces_examples=data_stream.produces_examples,
...             **kwargs)
... 
...     def transform_any(self, data):
...         if 'features' in self.sources:
...             data = list(data)
...             index = self.sources.index('features')
...             data[index] *= 2
...             data = tuple(data)
...         return data

So far so good, but our transformer applies the same transformation to every
source in the dataset. What if we want to be more general and be able to select
which sources we want to process with our transformer?

Transformers which are applied sourcewise like our doubler should usually
subclass from :class:`SourcewiseTransformer`. Their constructor takes an
additional ``which_sources`` keyword argument specifying which sources to apply
the transformer to. It's expected to be a tuple of source names. If
``which_sources`` is ``None``, then the transformer is applied to all sources.
Subclasses of :class:`SourcewiseTransformer` should implement a
:meth:`transform_source_example` method and/or a :meth:`transform_source_batch`
method, which apply on an individual source.

There also exists an :class:`AgnosticSourcewiseTransformer` class for cases
where the example and batch implementations of a sourcewise transformer are the
same. This class requires a :meth:`transform_any_source` method to be
implemented.

>>> from fuel.transformers import AgnosticSourcewiseTransformer
>>> 
>>> class Doubler(AgnosticSourcewiseTransformer):
...     def __init__(self, data_stream, **kwargs):
...         super(Doubler, self).__init__(
...             data_stream=data_stream,
...             produces_examples=data_stream.produces_examples,
...             **kwargs)
... 
...     def transform_any_source(self, source, _):
...         return 2 * source

Let's try this implementation on our dummy dataset.

>>> target_stream = Doubler(
...     data_stream=DataStream(
...         dataset=dataset,
...         iteration_scheme=batch_scheme),
...     which_sources=('targets',))
>>> all_stream = Doubler(
...     data_stream=DataStream(
...         dataset=dataset,
...         iteration_scheme=batch_scheme),
...     which_sources=None)
>>> print([batch for batch in target_stream.get_epoch_iterator()])
[(array([1, 2]), array([-2,  2])), (array([3, 4]), array([-2,  2]))]
>>> print([batch for batch in all_stream.get_epoch_iterator()])
[(array([2, 4]), array([-2,  2])), (array([6, 8]), array([-2,  2]))]

Finally, what if we not only want to select at runtime which sources the
transformation applies to, but also the transformer itself? This is what
the :class:`Mapping` transformer solves. In addition to a data stream, its
constructor accepts a ``mapping`` argument, which is expected to be a function.
This function which will be applied to data coming from the stream.

Here's how you would implement the feature doubler using :class:`Mapping`.

>>> from fuel.transformers import Mapping
>>> 
>>> features_index = dataset.sources.index('features')
>>> def double(data):
...     data = list(data)
...     data[features_index] *= 2
...     return tuple(data)
>>> mapping_stream = Mapping(
...     data_stream=DataStream(
...         dataset=dataset, iteration_scheme=batch_scheme),
...     mapping=double)
>>> print([batch for batch in mapping_stream.get_epoch_iterator()])
[(array([2, 4]), array([-1,  1])), (array([6, 8]), array([-1,  1]))]
