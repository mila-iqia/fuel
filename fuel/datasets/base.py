import collections
from abc import ABCMeta, abstractmethod

from six import add_metaclass

from picklable_itertools import iter_, izip

from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from fuel.utils import Subset


@add_metaclass(ABCMeta)
class Dataset(object):
    """A dataset.

    Dataset classes implement the interface to a particular dataset. The
    interface consists of a number of routines to manipulate so called
    "state" objects, e.g. open, reset and close them.

    Parameters
    ----------
    sources : tuple of strings, optional
        The data sources to load and return by :meth:`get_data`. By default
        all data sources are returned.
    axis_labels : dict, optional
        Maps source names to tuples of strings describing axis semantics,
        one per axis. Defaults to `None`, i.e. no information is available.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data e.g.
        ``('features',)`` when querying only the data from MNIST.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide e.g. ``('features',
        'targets')`` for MNIST (regardless of which data the data stream
        actually requests). Any implementation of a dataset should set this
        attribute on the class (or at least before calling ``super``).
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    default_transformers: It is expected to be a tuple with one element per
        transformer in the pipeline. Each element is a tuple with three
        elements:
            - the Transformer subclass to apply,
            - a list of arguments to pass to the subclass constructor, and
            - a dict of keyword arguments to pass to the subclass
              constructor.


    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.

    """
    provides_sources = None
    default_transformers = tuple()

    def __init__(self, sources=None, axis_labels=None):
        if not self.provides_sources:
            raise ValueError("dataset does not have `provides_sources`")
        if sources is not None:
            if not sources or not all(source in self.provides_sources
                                      for source in sources):
                raise ValueError("unable to provide requested sources")
            self.sources = sources
        self.axis_labels = axis_labels

    @property
    def sources(self):
        if not hasattr(self, '_sources'):
            return self.provides_sources
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

    def apply_default_transformers(self, stream):
        """Applies default transformers to a stream.

        Parameters
        ----------
        stream : :class:`~.streams.AbstractDataStream`
            A data stream.

        """
        for (cls, args, kwargs) in self.default_transformers:
            args = [stream] + args
            stream = cls(*args, **kwargs)
        return stream

    @property
    def example_iteration_scheme(self):
        if not hasattr(self, '_example_iteration_scheme'):
            raise AttributeError("dataset does not provide an example "
                                 "iteration scheme")
        return self._example_iteration_scheme

    @example_iteration_scheme.setter
    def example_iteration_scheme(self, value):
        self._example_iteration_scheme = value

    def get_example_stream(self):
        return DataStream(self, iteration_scheme=self.example_iteration_scheme)

    def open(self):
        """Return the state if the dataset requires one.

        Datasets which e.g. read files from disks require open file
        handlers, and this sort of stateful information should be handled
        by the data stream.

        Returns
        -------
        state : object
            An object representing the state of a dataset.

        """
        pass

    def reset(self, state):
        """Resets the state.

        Parameters
        ----------
        state : object
            The current state.

        Returns
        -------
        state : object
            A reset state.

        Notes
        -----
        The default implementation closes the state and opens a new one. A
        more efficient implementation (e.g. using ``file.seek(0)`` instead
        of closing and re-opening the file) can override the default one in
        derived classes.

        """
        self.close(state)
        return self.open()

    def next_epoch(self, state):
        """Switches the dataset state to the next epoch.

        The default implementation for this method is to reset the state.

        Parameters
        ----------
        state : object
            The current state.

        Returns
        -------
        state : object
            The state for the next epoch.

        """
        return self.reset(state)

    def close(self, state):
        """Cleanly close the dataset e.g. close file handles.

        Parameters
        ----------
        state : object
            The current state.

        """
        pass

    @abstractmethod
    def get_data(self, state=None, request=None):
        """Request data from the dataset.

        .. todo::

           A way for the dataset to communicate which kind of requests it
           accepts, and a way to communicate what kind of request is being
           sent when supporting multiple.

        Parameters
        ----------
        state : object, optional
            The state as returned by the :meth:`open` method. The dataset
            can use this to e.g. interact with files when needed.
        request : object, optional
            If supported, the request for a particular part of the data
            e.g. the number of examples to return, or the indices of a
            particular minibatch of examples.

        Returns
        -------
        tuple
            A tuple of data matching the order of :attr:`sources`.

        """

    def filter_sources(self, data):
        """Filter the requested sources from those provided by the dataset.

        A dataset can be asked to provide only a subset of the sources it
        can provide (e.g. asking MNIST only for the features, not for the
        labels). A dataset can choose to use this information to e.g. only
        load the requested sources into memory. However, in case the
        performance gain of doing so would be negligible, the dataset can
        load all the data sources and then use this method to return only
        those requested.

        Parameters
        ----------
        data : tuple of objects
            The data from all the sources i.e. should be of the same length
            as :attr:`provides_sources`.

        Returns
        -------
        tuple
            A tuple of data matching :attr:`sources`.

        Examples
        --------
        >>> import numpy
        >>> class Random(Dataset):
        ...     provides_sources = ('features', 'targets')
        ...     def get_data(self, state=None, request=None):
        ...         data = (numpy.random.rand(10), numpy.random.randn(3))
        ...         return self.filter_sources(data)
        >>> Random(sources=('targets',)).get_data() # doctest: +SKIP
        (array([-1.82436737,  0.08265948,  0.63206168]),)

        """
        return tuple([d for d, s in zip(data, self.provides_sources)
                      if s in self.sources])


class IterableDataset(Dataset):
    """Creates a dataset from a set of iterables.

    Parameters
    ----------
    iterables : :class:`~collections.OrderedDict` or iterable
        The iterable(s) to provide interface to. The iterables' `__iter__`
        method should return a new iterator over the iterable. If an
        :class:`~collections.OrderedDict` is given, its values should be
        iterables providing data, and its keys strings that are used as
        source names. If a single iterable is given, it will be given the
        source ``data``.

    Attributes
    ----------
    iterables : list
        A list of :class:`~collections.Iterable` objects.

    Notes
    -----
    Internally, this method uses picklable iterools's ``_iter``
    function, providing picklable alternatives to some iterators such as
    :func:`range`, :func:`tuple`, and even :class:`file`. However, if the
    iterable returns a different kind of iterator that is not picklable,
    you might want to consider using the :func:`.do_not_pickle_attributes`
    decorator.

    To iterate over a container in batches, combine this dataset with the
    :class:`Batch` data stream.

    """
    example_iteration_scheme = None

    def __init__(self, iterables, **kwargs):
        if isinstance(iterables, dict):
            self.provides_sources = tuple(iterables.keys())
        else:
            self.provides_sources = ('data',)
        super(IterableDataset, self).__init__(**kwargs)
        if isinstance(iterables, dict):
            if not all(isinstance(iterable, collections.Iterable)
                       for iterable in iterables.values()):
                raise ValueError
            self.iterables = [iterables[source] for source in self.sources]
        else:
            if not isinstance(iterables, collections.Iterable):
                raise ValueError
            self.iterables = [iterables]
        try:
            if len(set(len(iterable) for iterable in self.iterables)) != 1:
                raise ValueError("iterables are of different length")
        except TypeError:
            pass

    @property
    def num_examples(self):
        try:
            num_examples, = set(len(iterable) for iterable in self.iterables)
            return num_examples
        except TypeError:
            return float('nan')

    def open(self):
        iterators = [iter_(channel) for channel in self.iterables]
        return izip(*iterators)

    def get_data(self, state=None, request=None):
        if state is None or request is not None:
            raise ValueError
        return next(state)


class IndexableDataset(Dataset):
    """Creates a dataset from a set of indexable containers.

    Parameters
    ----------
    indexables : :class:`~collections.OrderedDict` or indexable
        The indexable(s) to provide interface to. This means it must
        support the syntax ```indexable[0]``. If an
        :class:`~collections.OrderedDict` is given, its values should be
        indexables providing data, and its keys strings that are used as
        source names. If a single indexable is given, it will be given the
        source ``data``.

    Attributes
    ----------
    indexables : list
        A list of indexable objects.

    Notes
    -----
    If the indexable data is very large, you might want to consider using
    the :func:`.do_not_pickle_attributes` decorator to make sure the data
    doesn't get pickled with the dataset, but gets reloaded/recreated
    instead.

    This dataset also uses the source names to create properties that
    provide easy access to the data.

    """
    def __init__(self, indexables, start=None, stop=None, **kwargs):
        if isinstance(indexables, dict):
            self.provides_sources = tuple(indexables.keys())
        else:
            self.provides_sources = ('data',)
        super(IndexableDataset, self).__init__(**kwargs)
        if isinstance(indexables, dict):
            self.indexables = [indexables[source][start:stop]
                               for source in self.sources]
            if not all(len(indexable) == len(self.indexables[0])
                       for indexable in self.indexables):
                raise ValueError("sources have different lengths")
        else:
            self.indexables = [indexables]

        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)

        self.start = start
        self.stop = stop
        self.subset = Subset(slice(start, stop), self.num_examples)

    def __getattr__(self, attr):
        if (attr not in ['sources', 'indexables', '_sources'] and
                attr in self.sources):
            return self.indexables[self.sources.index(attr)]
        raise AttributeError

    # Without explicitly defining a trivial __setstate__ method,
    # the __getattribute__ method would call the __getattr__ method,
    # which would raise an AttributeError. This causes problems
    # when unpickling.
    def __setstate__(self, dict):
        self.__dict__ = dict

    @property
    def num_examples(self):
        return len(self.indexables[0])

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        return tuple(self.subset.index_within_subset(indexable, request)
                     for indexable in self.indexables)
