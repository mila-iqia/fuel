from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
from multiprocessing import Process, Queue

import numpy
from picklable_itertools import chain, ifilter, izip
from six import add_metaclass, iteritems

from fuel import config
from fuel.streams import AbstractDataStream
from fuel.schemes import BatchSizeScheme
from ..exceptions import AxisLabelsMismatchError

log = logging.getLogger(__name__)


class ExpectsAxisLabels(object):
    """Mixin for transformers, used to verify axis labels.

    Notes
    -----
    Provides a method :meth:`verify_axis_labels` that should be called
    with the expected and actual values for an axis labels tuple. If
    `actual` is `None`, a warning is logged; if it is non-`None` and does
    not match `expected`, a :class:`AxisLabelsMismatchError` is raised.

    The check is only performed on the first call; if the call succeeds,
    an attribute is written to skip further checks, in the interest of
    speed.

    """
    def verify_axis_labels(self, expected, actual, source_name):
        """Verify that axis labels for a given source are as expected.

        Parameters
        ----------
        expected : tuple
            A tuple of strings representing the expected axis labels.
        actual : tuple or None
            A tuple of strings representing the actual axis labels, or
            `None` if they could not be determined.
        source_name : str
            The name of the source being checked. Used for caching the
            results of checks so that the check is only performed once.

        Notes
        -----
        Logs a warning in case of `actual=None`, raises an error on
        other mismatches.

        """
        if not getattr(self, '_checked_axis_labels', False):
            self._checked_axis_labels = defaultdict(bool)
        if not self._checked_axis_labels[source_name]:
            if actual is None:
                log.warning("%s instance could not verify (missing) axis "
                            "expected %s, got None",
                            self.__class__.__name__, expected)
            else:
                if expected != actual:
                    raise AxisLabelsMismatchError("{} expected axis labels "
                                                  "{}, got {} instead".format(
                                                      self.__class__.__name__,
                                                      expected, actual))
            self._checked_axis_labels[source_name] = True


@add_metaclass(ABCMeta)
class Transformer(AbstractDataStream):
    """A data stream that wraps another data stream.

    Subclasses must define a `transform_batch` method (to act on batches),
    a `transform_example` method (to act on individual examples), or
    both methods.

    Typically (using the interface mentioned above), the transformer
    is expected to have the same output type (example or batch) as its
    input type.  If the transformer subclass is going from batches to
    examples or vice versa, it should override `get_data` instead.
    Overriding `get_data` is also necessary when access to `request` is
    necessary (e.g. for the :class:`Cache` transformer).

    Attributes
    ----------
    child_epoch_iterator : iterator type
        When a new epoch iterator is requested, a new epoch creator is
        automatically requested from the wrapped data stream and stored in
        this attribute. Use it to access data from the wrapped data stream
        by calling ``next(self.child_epoch_iterator)``.
    produces_examples : bool
        Whether this transformer produces examples (as opposed to batches
        of examples).

    """
    def __init__(self, data_stream, produces_examples=None, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        if produces_examples is not None:
            self.produces_examples = produces_examples
        self.data_stream = data_stream

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.data_stream.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.data_stream.close()

    def reset(self):
        self.data_stream.reset()

    def next_epoch(self):
        self.data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the wrapped data set.

        Notes
        -----
        This default implementation assumes that the epochs of the wrapped
        data stream are less or equal in length to the original data
        stream. Implementations for which this is not true should request
        new epoch iterators from the child data set when necessary.

        """
        self.child_epoch_iterator = self.data_stream.get_epoch_iterator()
        return super(Transformer, self).get_epoch_iterator(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)

        if self.produces_examples != self.data_stream.produces_examples:
            types = {True: 'examples', False: 'batches'}
            raise NotImplementedError(
                "the wrapped data stream produces {} while the {} transformer "
                "produces {}, which it does not support.".format(
                    types[self.data_stream.produces_examples],
                    self.__class__.__name__,
                    types[self.produces_examples]))
        elif self.produces_examples:
            return self.transform_example(data)
        else:
            return self.transform_batch(data)

    def transform_example(self, example):
        """Transforms a single example."""
        raise NotImplementedError(
            "`{}` does not support examples as input, but the wrapped data "
            "stream produces examples.".format(self.__class__.__name__))

    def transform_batch(self, batch):
        """Transforms a batch of examples."""
        raise NotImplementedError(
            "`{}` does not support batches as input, but the wrapped data "
            "stream produces batches.".format(self.__class__.__name__))


@add_metaclass(ABCMeta)
class AgnosticTransformer(Transformer):
    """A transformer that operates the same on examples or batches.

    Subclasses must implement the `transform_any` method, which is to be
    applied to both examples and batches. This is useful when the example
    and batch implementation of a transformation are the same.

    """
    @abstractmethod
    def transform_any(self, data):
        """Transforms the input, which can either be an example or a batch."""

    def transform_example(self, example):
        return self.transform_any(example)

    def transform_batch(self, batch):
        return self.transform_any(batch)


class Mapping(Transformer):
    """Applies a mapping to the data of the wrapped data stream.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    mapping : callable
        The mapping to be applied.
    add_sources : tuple of str, optional
        When given, the data produced by the mapping is added to original
        data under source names `add_sources`.

    """
    def __init__(self, data_stream, mapping, add_sources=None, **kwargs):
        super(Mapping, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.mapping = mapping
        self.add_sources = add_sources

    @property
    def sources(self):
        return self.data_stream.sources + (self.add_sources
                                           if self.add_sources else ())

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        image = self.mapping(data)
        if not self.add_sources:
            return image
        return data + image


@add_metaclass(ABCMeta)
class SourcewiseTransformer(Transformer):
    """Applies a transformation sourcewise.

    Subclasses must define `transform_source_example` (to transform
    examples), `transform_source_batch` (to transform batches) or
    both.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.

    """
    def __init__(self, data_stream, produces_examples, which_sources=None,
                 **kwargs):
        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources
        super(SourcewiseTransformer, self).__init__(
            data_stream, produces_examples, **kwargs)

    def _apply_sourcewise_transformation(self, data, method):
        data = list(data)
        for i, source_name in enumerate(self.data_stream.sources):
            if source_name in self.which_sources:
                data[i] = method(data[i], source_name)
        return tuple(data)

    def transform_source_example(self, source_example, source_name):
        """Applies a transformation to an example from a source.

        Parameters
        ----------
        source_example : :class:`numpy.ndarray`
            An example from a source.
        source_name : str
            The name of the source being operated upon.

        """
        raise NotImplementedError(
            "`{}` does not support examples as input, but the wrapped data "
            "stream produces examples.".format(self.__class__.__name__))

    def transform_source_batch(self, source_batch, source_name):
        """Applies a transformation to a batch from a source.

        Parameters
        ----------
        source_batch : :class:`numpy.ndarray`
            A batch of examples from a source.
        source_name : str
            The name of the source being operated upon.

        """
        raise NotImplementedError(
            "`{}` does not support batches as input, but the wrapped data "
            "stream produces batches.".format(self.__class__.__name__))

    def transform_example(self, example):
        return self._apply_sourcewise_transformation(
            data=example, method=self.transform_source_example)

    def transform_batch(self, batch):
        return self._apply_sourcewise_transformation(
            data=batch, method=self.transform_source_batch)


@add_metaclass(ABCMeta)
class AgnosticSourcewiseTransformer(AgnosticTransformer,
                                    SourcewiseTransformer):
    """A sourcewise transformer that operates the same on examples or batches.

    Subclasses must implement the `transform_any_source` method, which is
    to be applied to both examples and batches. This is useful when the
    example and batch implementation of a sourcewise transformation are
    the same.

    """
    def transform_any(self, data):
        return self._apply_sourcewise_transformation(
            data=data, method=self.transform_any_source)

    @abstractmethod
    def transform_any_source(self, source_data, source_name):
        """Applies a transformation to a source.

        The data can either be an example or a batch of examples.

        Parameters
        ----------
        source_data : :class:`numpy.ndarray`
            Data from a source.
        source_name : str
            The name of the source being operated upon.

        """


class Flatten(SourcewiseTransformer):
    """Flattens selected sources.

    If the wrapped data stream produces batches, they will be flattened
    along all but the first axis.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    """
    def __init__(self, data_stream, **kwargs):
        # Modify the axis_labels dict to reflect the fact that all non-batch
        # axes will be grouped together under the same 'feature' axis.
        if data_stream.axis_labels:
            which_sources = kwargs.get('which_sources', data_stream.sources)
            kwargs.setdefault(
                'axis_labels',
                self._infer_axis_labels(data_stream, which_sources))
        super(Flatten, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def _infer_axis_labels(self, data_stream, which_sources):
        axis_labels = {}
        for source, labels in iteritems(data_stream.axis_labels):
            if source in which_sources:
                if not labels:
                    axis_labels[source] = None
                elif data_stream.produces_examples:
                    axis_labels[source] = ('feature',)
                else:
                    axis_labels[source] = (labels[0], 'feature')
            else:
                axis_labels[source] = labels
        return axis_labels

    def transform_source_example(self, source_example, _):
        return numpy.asarray(source_example).flatten()

    def transform_source_batch(self, source_batch, _):
        return numpy.asarray(source_batch).reshape((len(source_batch), -1))


class ScaleAndShift(AgnosticSourcewiseTransformer):
    """Scales and shifts selected sources by scalar quantities.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    Parameters
    ----------
    scale : float
        Scaling factor.
    shift : float
        Shifting factor.

    """
    def __init__(self, data_stream, scale, shift, **kwargs):
        self.scale = scale
        self.shift = shift
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(ScaleAndShift, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(source_data) * self.scale + self.shift


class Cast(AgnosticSourcewiseTransformer):
    """Casts selected sources as some dtype.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    Parameters
    ----------
    dtype : str
        Data type to cast to. Can be any valid numpy dtype, or 'floatX',
        in which case ``fuel.config.floatX`` is used.

    """
    def __init__(self, data_stream, dtype, **kwargs):
        if dtype == 'floatX':
            dtype = config.floatX
        self.dtype = dtype
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(Cast, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(source_data, dtype=self.dtype)


class ForceFloatX(AgnosticSourcewiseTransformer):
    """Force all floating point numpy arrays to be floatX."""
    def __init__(self, data_stream, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(ForceFloatX, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        source_needs_casting = (isinstance(source_data, numpy.ndarray) and
                                source_data.dtype.kind == "f" and
                                source_data.dtype != config.floatX)
        if source_needs_casting:
            source_data = source_data.astype(config.floatX)
        return source_data


class Filter(Transformer):
    """Filters samples that meet a predicate.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The filtered data stream.
    predicate : callable
        Should return ``True`` for the samples to be kept.

    """
    def __init__(self, data_stream, predicate, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(Filter, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.predicate = predicate

    def get_epoch_iterator(self, **kwargs):
        super(Filter, self).get_epoch_iterator(**kwargs)
        return ifilter(self.predicate, self.child_epoch_iterator)


class Cache(Transformer):
    """Cache examples when sequentially reading a dataset.

    Given a data stream which reads large chunks of data, this data
    stream caches these chunks and returns smaller batches from it until
    exhausted.

    Parameters
    ----------
    iteration_scheme : :class:`.IterationScheme`
        Note that this iteration scheme must return batch sizes (integers),
        which must necessarily be smaller than the child data stream i.e.
        the batches returned must be smaller than the cache size.

    Attributes
    ----------
    cache : list of lists of objects
        This attribute holds the cache at any given point. It is a list of
        the same size as the :attr:`sources` attribute. Each element in
        this list in its turn a list of examples that are currently in the
        cache. The cache gets emptied at the start of each epoch, and gets
        refilled when needed through the :meth:`get_data` method.

    """
    def __init__(self, data_stream, iteration_scheme, **kwargs):
        # Note: produces_examples will always be False because of this
        # restriction: the only iteration schemes allowed are BatchSizeScheme,
        # which produce batches.
        if not isinstance(iteration_scheme, BatchSizeScheme):
            raise ValueError('iteration scheme must be an instance of '
                             'BatchSizeScheme')
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(Cache, self).__init__(
            data_stream, iteration_scheme=iteration_scheme, **kwargs)
        self.cache = [[] for _ in self.sources]

    def get_data(self, request=None):
        if request is None:
            raise ValueError
        if request > len(self.cache[0]):
            self._cache()
        data = []
        for i, cache in enumerate(self.cache):
            data.append(numpy.asarray(cache[:request]))
            self.cache[i] = cache[request:]
        return tuple(data)

    def get_epoch_iterator(self, **kwargs):
        self.cache = [[] for _ in self.sources]
        return super(Cache, self).get_epoch_iterator(**kwargs)

    def _cache(self):
        try:
            for cache, data in zip(self.cache,
                                   next(self.child_epoch_iterator)):
                cache.extend(data)
        except StopIteration:
            if not self.cache[0]:
                raise


class SortMapping(object):
    """Callable class for creating sorting mappings.

    This class can be used to create a callable that can be used by the
    :class:`Mapping` constructor.

    Parameters
    ----------
    key : callable
        The mapping that returns the value to sort on. Its input will be
        a tuple that contains a single data point for each source.
    reverse : boolean value that indicates whether the sort order should
        be reversed.

    """
    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def __call__(self, batch):
        output = sorted(zip(*batch), key=self.key, reverse=self.reverse)
        output = tuple(numpy.asarray(i) if isinstance(j, numpy.ndarray)
                       else list(i)
                       for i, j in zip(zip(*output), batch))
        return output


class Batch(Transformer):
    """Creates minibatches from data streams providing single examples.

    Some datasets only return one example at at time e.g. when reading text
    files a line at a time. This wrapper reads several examples
    sequentially to turn those into minibatches.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap.
    iteration_scheme : :class:`.BatchSizeScheme` instance
        The iteration scheme to use; should return integers representing
        the size of the batch to return.
    strictness : int, optional
        How strictly the iterator should adhere to the batch size. By
        default, the value 0 means that the last batch is returned
        regardless of its size, so it can be smaller than what is actually
        requested. At level 1, the last batch is discarded if it is not of
        the correct size. At the highest strictness level, 2, an error is
        raised if a batch of the requested size cannot be provided.

    """
    def __init__(self, data_stream, iteration_scheme, strictness=0, **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        # The value for `produces_examples` is inferred from the iteration
        # scheme's `requests_examples` attribute. We expect the scheme to
        # request batches.
        if iteration_scheme.requests_examples:
            raise ValueError('the iteration scheme must request batches, '
                             'not individual examples.')
        if data_stream.axis_labels:
            kwargs.setdefault(
                'axis_labels',
                dict((source, ('batch',) + labels if labels else None) for
                     source, labels in iteritems(data_stream.axis_labels)))
        super(Batch, self).__init__(
            data_stream, iteration_scheme=iteration_scheme, **kwargs)
        self.strictness = strictness

    def get_data(self, request=None):
        """Get data from the dataset."""
        if request is None:
            raise ValueError
        data = [[] for _ in self.sources]
        for i in range(request):
            try:
                for source_data, example in zip(
                        data, next(self.child_epoch_iterator)):
                    source_data.append(example)
            except StopIteration:
                # If some data has been extracted and `strict` is not set,
                # we should spit out this data before stopping iteration.
                if not self.strictness and data[0]:
                    break
                elif self.strictness > 1 and data[0]:
                    raise ValueError
                raise
        return tuple(numpy.asarray(source_data) for source_data in data)


class Unpack(Transformer):
    """Unpacks batches to compose a stream of examples.

    This class is the inverse of the Batch class: it turns a minibatch into
    a stream of examples.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to unpack

    """
    def __init__(self, data_stream, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        if data_stream.axis_labels:
            kwargs.setdefault(
                'axis_labels',
                dict((source, labels[1:] if labels else None) for
                     source, labels in iteritems(data_stream.axis_labels)))
        super(Unpack, self).__init__(
            data_stream, produces_examples=True, **kwargs)
        self.data = None

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.data:
            data = next(self.child_epoch_iterator)
            self.data = izip(*data)
        try:
            return next(self.data)
        except StopIteration:
            self.data = None
            return self.get_data()


class Padding(Transformer):
    """Adds padding to variable-length sequences.

    When your batches consist of variable-length sequences, use this class
    to equalize lengths by adding zero-padding. To distinguish between
    data and padding masks can be produced. For each data source that is
    masked, a new source will be added. This source will have the name of
    the original source with the suffix ``_mask`` (e.g. ``features_mask``).

    Elements of incoming batches will be treated as numpy arrays (i.e.
    using `numpy.asarray`). If they have more than one dimension,
    all dimensions except length, that is the first one, must be equal.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    mask_sources : tuple of strings, optional
        The sources for which we need to add a mask. If not provided, a
        mask will be created for all data sources
    mask_dtype: str, optional
        data type of masks. If not provided, floatX from config will
        be used.

    """
    def __init__(self, data_stream, mask_sources=None, mask_dtype=None,
                 **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        super(Padding, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        if mask_sources is None:
            mask_sources = self.data_stream.sources
        self.mask_sources = mask_sources
        if mask_dtype is None:
            self.mask_dtype = config.floatX
        else:
            self.mask_dtype = mask_dtype

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.mask_sources:
                sources.append(source + '_mask')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_batch[0]).dtype

            padded_batch = numpy.zeros(
                (len(source_batch), max_sequence_length) + rest_shape,
                dtype=dtype)
            for i, sample in enumerate(source_batch):
                padded_batch[i, :len(sample)] = sample
            batch_with_masks.append(padded_batch)

            mask = numpy.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class Merge(AbstractDataStream):
    """Merges several datastreams into a single one.

    Parameters
    ----------
    data_streams : iterable
        The data streams to merge.
    sources : iterable
        A collection of strings, determining what sources should be called.

    Examples
    --------
    >>> from fuel.datasets import IterableDataset
    >>> english = IterableDataset(['Hello world!'])
    >>> french = IterableDataset(['Bonjour le monde!'])
    >>> from fuel.streams import DataStream
    >>> streams = (DataStream(english),
    ...            DataStream(french))
    >>> merged_stream = Merge(streams, ('english', 'french'))
    >>> merged_stream.sources
    ('english', 'french')
    >>> next(merged_stream.get_epoch_iterator())
    ('Hello world!', 'Bonjour le monde!')

    """
    def __init__(self, data_streams, sources, axis_labels=None):
        super(Merge, self).__init__(
            iteration_scheme=None, axis_labels=axis_labels)
        if not all(data_stream.produces_examples ==
                   data_streams[0].produces_examples
                   for data_stream in data_streams):
            raise ValueError('all data streams must produce the same type of '
                             'output (batches or examples)')
        self.data_streams = data_streams
        self.produces_examples = self.data_streams[0].produces_examples

        if len(list(chain(*[data_stream.sources for data_stream
                            in data_streams]))) != len(sources):
            raise ValueError("wrong number of sources given")
        self.sources = sources

    def close(self):
        for data_stream in self.data_streams:
            data_stream.close()

    def reset(self):
        for data_stream in self.data_streams:
            data_stream.reset()

    def next_epoch(self):
        for data_stream in self.data_streams:
            data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        self.child_epoch_iterators = [data_stream.get_epoch_iterator()
                                      for data_stream in self.data_streams]
        return super(Merge, self).get_epoch_iterator(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        result = []
        for child_epoch_iterator in self.child_epoch_iterators:
            result.extend(next(child_epoch_iterator))
        return tuple(result)


class BackgroundProcess(object):
    """A background process that reads batches and stores them in a queue.

    The :meth:`main` method needs to be called in order to start reading
    batches into the queue. Note that this process will run infinitely;
    start it as a :attr:`~multiprocessing.Process.daemon` to make sure it
    will get killed when the main process exits.

    Parameters
    ----------
    data_stream : :class:`.DataStream` or :class:`Transformer`
        The data stream from which to read batches.
    max_batches : int
        The maximum number of batches to store in the queue. If reached,
        the process wil block until a batch is popped from the queue.

    """
    def __init__(self, data_stream, max_batches):
        self.data_stream = data_stream
        self.batches = Queue(max_batches)
        self.run_background = True

    def main(self):
        while True:
            iterator = self.data_stream.get_epoch_iterator()
            for batch in iterator:
                self.batches.put(batch)
            self.batches.put(StopIteration)

    def get_next_data(self):
        return self.batches.get()


class MultiProcessing(Transformer):
    """Cache batches from the stream in a separate process.

    To speed up training of your model, it can be worthwhile to load and
    process data in separate process. This is a simple implementation of
    such an approach that makes use of Python's :mod:`multiprocessing`
    module.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`
        The data stream to read batches from in the separate process.
    max_store : int, optional
        The maximum number of batches to keep in the queue.

    Notes
    -----
    This approach incurs an overhead from the need to serialize batches in
    order to send them to the main process. This should be acceptable if
    your model's training calls take significantly longer than reading a
    batch of data does, but for fast models or slow data pipelines a more
    robust approach might need to be considered.

    """
    def __init__(self, data_stream, max_store=100, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(MultiProcessing, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.background = BackgroundProcess(data_stream, max_store)
        self.proc = Process(target=self.background.main)
        self.proc.daemon = True
        self.proc.start()

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = self.background.get_next_data()
        if data == StopIteration:
            raise StopIteration
        return data


class Rename(AgnosticTransformer):
    """Renames the sources of the stream.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    names : dict
        A dictionary mapping the old and new names of the sources
        to rename.
    on_non_existent : str, optional
        Desired behaviour when a source specified as a key in `names`
        is not provided by the streams: see `on_overwrite` above for
        description of possible values. Default is 'raise'.

    """
    def __init__(self, data_stream, names, on_non_existent='raise', **kwargs):
        if on_non_existent not in ('raise', 'ignore', 'warn'):
            raise ValueError("on_non_existent must be one of 'raise', "
                             "'ignore', 'warn'")
        # We allow duplicate values in the full dictionary, but those
        # that correspond to keys that are real sources in the data stream
        # must be unique. This lets you use one piece of code including
        # a Rename transformer to map disparately named sources in
        # different datasets to a common name.
        usable_names = {k: v for k, v in iteritems(names)
                        if k in data_stream.sources}
        if len(set(usable_names.values())) != len(usable_names):
            raise KeyError("multiple old source names cannot map to "
                           "the same new source name")
        sources = list(data_stream.sources)
        sources_lookup = {n: i for i, n in enumerate(sources)}
        for old, new in iteritems(names):
            if new in sources_lookup and new not in names:
                if old in usable_names:
                    message = ("Renaming source '{}' to '{}' "
                               "would create two sources named '{}'"
                               .format(old, new, new))
                    raise KeyError(message)
            if old not in sources_lookup:
                message = ("Renaming source '{}' to '{}': "
                           "stream does not provide a source '{}'"
                           .format(old, new, old))
                if on_non_existent == 'raise':
                    raise KeyError(message)
                else:
                    log_level = {'warn': logging.WARNING,
                                 'ignore': logging.DEBUG}
                    log.log(log_level[on_non_existent], message)
            else:
                sources[sources_lookup[old]] = new
        self.sources = tuple(sources)
        if data_stream.axis_labels:
            kwargs.setdefault(
                'axis_labels',
                dict((names[source] if source in names else source, labels)
                     for (source, labels) in
                     iteritems(data_stream.axis_labels)))
        super(Rename, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any(self, data):
        return data


class FilterSources(AgnosticTransformer):
    """Selects a subset of the stream sources.

    Order of data stream's sources is maintained. The order of sources
    given as parameter to FilterSources does not matter.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` or :class:`Transformer`.
        The data stream.
    sources : tuple of strings
        The names of the data sources returned by this transformer.
        Must be a subset of the sources given by the stream.

    """
    def __init__(self, data_stream, sources, **kwargs):
        if any(source not in data_stream.sources for source in sources):
            raise ValueError("sources must all be contained in "
                             "data_stream.sources")
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels',
                              dict((source, labels) for (source, labels)
                                   in iteritems(data_stream.axis_labels)
                                   if source in sources))
        super(FilterSources, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

        # keep order of data_stream.sources
        self.sources = tuple(s for s in data_stream.sources if s in sources)

    def transform_any(self, data):
        return [d for d, s in izip(data, self.data_stream.sources)
                if s in self.sources]
