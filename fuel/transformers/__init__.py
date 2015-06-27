from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Queue

import numpy
from picklable_itertools import chain, ifilter, izip, imap, repeat, starmap
from picklable_itertools.extras import partition
from six import add_metaclass, iteritems

from fuel import config
from fuel.streams import AbstractDataStream


@add_metaclass(ABCMeta)
class Transformer(AbstractDataStream):
    """A data stream that wraps another data stream.

    Attributes
    ----------
    child_epoch_iterator : iterator type
        When a new epoch iterator is requested, a new epoch creator is
        automatically requested from the wrapped data stream and stored in
        this attribute. Use it to access data from the wrapped data stream
        by calling ``next(self.child_epoch_iterator)``.
    batch_input : boolean
        Specification whether the input stream
        is working on example or batch

    """
    def __init__(self, data_stream, batch_input=False, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.data_stream = data_stream
        self.batch_input = batch_input

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
        if self.batch_input:
            return self.get_data_from_batch(request)
        else:
            return self.get_data_from_example(request)

    def get_data_from_example(self, request=None):
        raise NotImplementedError(
            "`{}` does not support examples as inputs, "
            "but `batch_input` was set to `False`".format(type(self))
        )

    def get_data_from_batch(self, request=None):
        raise NotImplementedError(
            "`{}` does not support batches as inputs, "
            "but `batch_input` was set to `False`".format(type(self))
        )


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
    def __init__(self, data_stream, mapping, add_sources=None):
        super(Mapping, self).__init__(data_stream)
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
class SingleMapping(Transformer):
    """Applies a single mapping to multiple sources.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.

    """
    def __init__(self, data_stream, which_sources=None):
        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources
        super(SingleMapping, self).__init__(data_stream)

    @abstractmethod
    def mapping(self, source):
        """Applies a single mapping to selected sources.

        Parameters
        ----------
        source : :class:`numpy.ndarray`
            Input source.

        """

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        for i, source_name in enumerate(self.data_stream.sources):
            if source_name in self.which_sources:
                data[i] = self.mapping(data[i])
        return tuple(data)


class Flatten(SingleMapping):
    """Flattens selected sources along all but the first axis."""
    def __init__(self, data_stream, **kwargs):
        super(Flatten, self).__init__(data_stream, **kwargs)

    def mapping(self, source):
        return source.reshape((source.shape[0], -1))


class ScaleAndShift(SingleMapping):
    """Scales and shifts selected sources by scalar quantities.

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
        super(ScaleAndShift, self).__init__(data_stream, **kwargs)

    def mapping(self, source):
        return source * self.scale + self.shift


class Cast(SingleMapping):
    """Casts selected sources as some dtype.

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
        super(Cast, self).__init__(data_stream, **kwargs)

    def mapping(self, source):
        return source.astype(self.dtype)


class ForceFloatX(Transformer):
    """Force all floating point numpy arrays to be floatX."""
    def __init__(self, data_stream):
        super(ForceFloatX, self).__init__(
            data_stream, axis_labels=data_stream.axis_labels)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        result = []
        for piece in data:
            if (isinstance(piece, numpy.ndarray) and
                    piece.dtype.kind == "f" and
                    piece.dtype != config.floatX):
                result.append(piece.astype(config.floatX))
            else:
                result.append(piece)
        return tuple(result)


class Filter(Transformer):
    """Filters samples that meet a predicate.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The filtered data stream.
    predicate : callable
        Should return ``True`` for the samples to be kept.

    """
    def __init__(self, data_stream, predicate):
        super(Filter, self).__init__(data_stream)
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
    def __init__(self, data_stream, iteration_scheme):
        super(Cache, self).__init__(
            data_stream, iteration_scheme=iteration_scheme, batch_input=True)
        self.cache = [[] for _ in self.sources]

    def get_data_from_batch(self, request=None):
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
        for cache, data in zip(self.cache, next(self.child_epoch_iterator)):
            cache.extend(data)


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
    def __init__(self, data_stream, iteration_scheme, strictness=0):
        super(Batch, self).__init__(
            data_stream, iteration_scheme=iteration_scheme)
        self.strictness = strictness

    def get_data_from_example(self, request=None):
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
    def __init__(self, data_stream):
        super(Unpack, self).__init__(data_stream, batch_input=True)
        self.data = None

    def get_data_from_batch(self, request=None):
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
    def __init__(self, data_stream, mask_sources=None, mask_dtype=None):
        super(Padding, self).__init__(data_stream, batch_input=True)
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

    def get_data_from_batch(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        data_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            if source not in self.mask_sources:
                data_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.zeros(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype)
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)
        return tuple(data_with_masks)


class Merge(Transformer):
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
    >>> streams = (english.get_example_stream(),
    ...            french.get_example_stream())
    >>> merged_stream = Merge(streams, ('english', 'french'))
    >>> merged_stream.sources
    ('english', 'french')
    >>> next(merged_stream.get_epoch_iterator())
    ('Hello world!', 'Bonjour le monde!')

    """
    def __init__(self, data_streams, sources):
        self.data_streams = data_streams
        if len(list(chain(*[data_stream.sources for data_stream
                            in data_streams]))) != len(sources):
            raise ValueError("wrong number of sources given")
        self.sources = sources

    def get_epoch_iterator(self, **kwargs):
        batches = chain.from_iterable(
            izip(*[data_stream.get_epoch_iterator()
                   for data_stream in self.data_streams]))

        part = partition(len(self.sources), chain.from_iterable(batches))
        as_dict = kwargs.get('as_dict', False)
        if as_dict:
            return imap(dict, starmap(zip, izip(repeat(self.sources), part)))
        else:
            return part


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
    def __init__(self, data_stream, max_store=100):
        super(MultiProcessing, self).__init__(data_stream)
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


class Rename(Transformer):
    """Renames the sources of the stream.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    names : dict
        A dictionary mapping the old and new names of the sources
        to rename.

    """
    def __init__(self, data_stream, names):
        super(Rename, self).__init__(data_stream)
        sources = list(self.data_stream.sources)
        for old, new in iteritems(names):
            if old not in sources:
                raise KeyError("%s not in the sources of the stream" % old)
            else:
                sources[sources.index(old)] = new
        self.sources = tuple(sources)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        return data


class FilterSources(Transformer):
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
    def __init__(self, data_stream, sources):
        super(FilterSources, self).__init__(data_stream)
        if any(source not in data_stream.sources for source in sources):
            raise ValueError("sources must all be contained in "
                             "data_stream.sources")

        # keep order of data_stream.sources
        self.sources = tuple([s for s in data_stream.sources if s in sources])

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        data = next(self.child_epoch_iterator)
        return [d for d, s in izip(data, self.data_stream.sources)
                if s in self.sources]
