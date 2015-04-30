from abc import ABCMeta, abstractmethod
from collections import Iterable

import numpy
from picklable_itertools import chain, repeat, imap, iter_
from picklable_itertools.extras import partition_all
from six import add_metaclass
from six.moves import xrange

from fuel import config


@add_metaclass(ABCMeta)
class IterationScheme(object):
    """An iteration scheme.

    Iteration schemes provide a dataset-agnostic iteration scheme, such as
    sequential batches, shuffled batches, etc. for datasets that choose to
    support them.

    Notes
    -----
    Iteration schemes implement the :meth:`get_request_iterator` method,
    which returns an iterator type (e.g. a generator or a class which
    implements the `iterator protocol`_).

    Stochastic iteration schemes should generally not be shared between
    different data streams, because it would make experiments harder to
    reproduce.

    .. _iterator protocol:
       https://docs.python.org/3.3/library/stdtypes.html#iterator-types

    """
    @abstractmethod
    def get_request_iterator(self):
        raise NotImplementedError


@add_metaclass(ABCMeta)
class BatchSizeScheme(IterationScheme):
    """Iteration scheme that returns batch sizes.

    For infinite datasets it doesn't make sense to provide indices to
    examples, but the number of samples per batch can still be given.
    Hence BatchSizeScheme is the base class for iteration schemes
    that only provide the number of examples that should be in a batch.

    """
    pass


@add_metaclass(ABCMeta)
class BatchScheme(IterationScheme):
    """Iteration schemes that return slices or indices for batches.

    For datasets where the number of examples is known and easily
    accessible (as is the case for most datasets which are small enough
    to be kept in memory, like MNIST) we can provide slices or lists of
    labels to the dataset.

    Parameters
    ----------
    examples : int or list
        Defines which examples from the dataset are iterated.
        If list, its items are the indices of examples.
        If an integer, it will use that many examples from the beginning
        of the dataset, i.e. it is interpreted as range(examples)
    batch_size : int
        The request iterator will return slices or list of indices in
        batches of size `batch_size` until the end of `examples` is
        reached.
        Note that this means that the last batch size returned could be
        smaller than `batch_size`. If you want to ensure all batches are
        of equal size, then ensure len(`examples`) or `examples` is a
        multiple of `batch_size`.

    """
    def __init__(self, examples, batch_size):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)
        self.batch_size = batch_size


@add_metaclass(ABCMeta)
class IndexScheme(IterationScheme):
    """Iteration schemes that return single indices.

    This is for datasets that support indexing (like :class:`BatchScheme`)
    but where we want to return single examples instead of batches.

    """
    def __init__(self, examples):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)


class ConstantScheme(BatchSizeScheme):
    """Constant batch size iterator.

    This subset iterator simply returns the same constant batch size
    for a given number of times (or else infinitely).

    Parameters
    ----------
    batch_size : int
        The size of the batch to return.
    num_examples : int, optional
        If given, the request iterator will return `batch_size` until the
        sum reaches `num_examples`. Note that this means that the last
        batch size returned could be smaller than `batch_size`. If you want
        to ensure all batches are of equal size, then pass `times` equal to
        ``num_examples / batch-size`` instead.
    times : int, optional
        The number of times to return `batch_size`.

    """
    def __init__(self, batch_size, num_examples=None, times=None):
        if num_examples and times:
            raise ValueError
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.times = times

    def get_request_iterator(self):
        if self.times:
            return repeat(self.batch_size, self.times)
        if self.num_examples:
            d, r = divmod(self.num_examples, self.batch_size)
            return chain(repeat(self.batch_size, d), [r] if r else [])
        return repeat(self.batch_size)


class SequentialScheme(BatchScheme):
    """Sequential batches iterator.

    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def get_request_iterator(self):
        return imap(list, partition_all(self.batch_size, self.indices))


class ShuffledScheme(BatchScheme):
    """Shuffled batches iterator.

    Iterate over all the examples in a dataset of fixed size in shuffled
    batches.

    Parameters
    ----------
    sorted_indices : bool, optional
        If `True`, enforce that indices within a batch are ordered.
        Defaults to `False`.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    Shuffling the batches requires creating a shuffled list of indices in
    memory. This can be memory-intensive for very large numbers of examples
    (i.e. in the order of tens of millions).

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        self.sorted_indices = kwargs.pop('sorted_indices', False)
        super(ShuffledScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        self.rng.shuffle(indices)
        if self.sorted_indices:
            return imap(sorted, partition_all(self.batch_size, indices))
        else:
            return imap(list, partition_all(self.batch_size, indices))


class SequentialExampleScheme(IndexScheme):
    """Sequential examples iterator.

    Returns examples in order.

    """
    def get_request_iterator(self):
        return iter_(self.indices)


class ShuffledExampleScheme(IndexScheme):
    """Shuffled examples iterator.

    Returns examples in random order.

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        super(ShuffledExampleScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        self.rng.shuffle(indices)
        return iter_(indices)
