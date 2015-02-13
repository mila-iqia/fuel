from picklable_itertools import _iter, izip

from fuel.datasets import Dataset


class ContainerDataset(Dataset):
    """Equips a Python container with the dataset interface.

    Parameters
    ----------
    container : iterable
        The container to provide interface to. The container's `__iter__`
        method should return a new iterator over the container. If the
        container given is an instance of `dict` or `OrderedDict`, its
        values are interpreted as data channels and its keys are used as
        source names. Note, that only if the container is an OrderedDict
        the order of elements in the returned tuples is determined. If the
        iterable is not a dictionary, the source ``data`` will be used.

    Notes
    -----
    To iterate over a container in batches, combine this dataset with the
    :class:`BatchDataStream` data stream.

    """
    default_scheme = None

    def __init__(self, container, sources=None):
        if isinstance(container, dict):
            self.provides_sources = (sources if sources is not None
                                     else tuple(container.keys()))
            self.data_channels = [container[source] for source in self.sources]
        else:
            self.provides_sources = ('data',)
            if not (sources == self.sources or sources is None):
                raise ValueError
            self.data_channels = [container]

    def open(self):
        iterators = [_iter(channel) for channel in self.data_channels]
        return izip(*iterators)

    def get_data(self, state=None, request=None):
        if state is None or request is not None:
            raise ValueError
        return next(state)
