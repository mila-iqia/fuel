import h5py
import tables
from six import next
from six.moves import filter

from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('nodes')
class Hdf5Dataset(Dataset):
    """An HDF5 dataset.

    Parameters
    ----------
    sources : tuple of strings
        Sources which the dataset returns
    start : int
        Start index
    stop : int
        Stop index
    data_node : str
        Parent data node in HDF5 file
    sources_in_file : tuple of strings
        Names of nodes in HDF5 file which contain sources. Should the same
        length as `sources`.
        Optional, if not set will be equal to `sources`.

    """
    def __init__(self, sources, start, stop, path, data_node='Data',
                 sources_in_file=None):
        if sources_in_file is None:
            sources_in_file = sources
        self.sources_in_file = sources_in_file
        self.provides_sources = sources
        self.path = path
        self.data_node = data_node
        self.start = start
        self.stop = stop
        self.num_examples = self.stop - self.start
        self.nodes = None
        self.open_file(self.path)
        super(Hdf5Dataset, self).__init__(self.provides_sources)

    def open_file(self, path):
        h5file = tables.open_file(path, mode="r")
        node = h5file.getNode('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]

    def load(self):
        self.open_file(self.path)

    def get_data(self, state=None, request=None):
        """ Returns data from HDF5 dataset.

        .. note:: The best performance if `request` is a slice.

        """
        if self.start:
            if isinstance(request, slice):
                request = slice(request.start + self.start,
                                request.stop + self.start, request.step)
            elif isinstance(request, list):
                request = [index + self.start for index in request]
            else:
                raise ValueError
        data = [node[request] for node in self.nodes]
        return data


@do_not_pickle_attributes('data_sources')
class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * The dataset is not explicitly split. Instead, splits are defined as
      attributes of the root group. They're expected to be numpy arrays of
      shape (2,), with the first element being the starting point
      (inclusive) of the split and the last element being the stopping
      point (exclusive) of the split.

    The `which_set`, `start` and `stop` parameters work together in the
    following way:

    * `which_set` is resolved first. If it is `None`, the whole dataset is
      used.
    * `start` and `stop` define a slice *within the context of*
      `which_set`.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    which_set : str, optional
        Name of the root group attribute containing the split information.
        Defaults to `None`, in which case the whole dataset is used.
    start : int
        Start index *within the split*.
    stop : int
        Stop index *within the split*.
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.

    """
    ref_counts = dict()

    def __init__(self, path, which_set=None, start=None, stop=None,
                 load_in_memory=False, driver=None, **kwargs):
        self.path = path
        self.which_set = which_set
        self.start = start
        self.stop = stop
        self.load_in_memory = load_in_memory
        self.driver = driver
        self.load()
        super(H5PYDataset, self).__init__(**kwargs)

    def _get_file_id(self):
        try:
            return next(
                filter(lambda x: x.name == self.path, self.ref_counts.keys()))
        except StopIteration:
            return self.path

    def load(self):
        handle = self._out_of_memory_open()
        self.provides_sources = list(handle.keys())
        shapes = [data_source.shape for data_source in handle.values()]
        if any(s[0] != shapes[0][0] for s in shapes):
            raise ValueError("sources have different lengths")
        start, stop = (handle.attrs[self.which_set] if self.which_set
                       else (0, shapes[0][0]))
        self._start = start if self.start is None else start + self.start
        self._stop = stop if self.stop is None else start + self.stop
        self.num_examples = self._stop - self._start
        self.data_sources = ([data_source[self._start: self._stop] for
                              data_source in handle.values()]
                             if self.load_in_memory else None)
        self._out_of_memory_close(handle)

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        file_id = self._get_file_id()
        state = h5py.File(name=file_id, mode="r", driver=self.driver)
        self.ref_counts[state.id] = self.ref_counts.get(state.id, 0) + 1
        return state

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close(state)

    def _out_of_memory_close(self, state):
        self.ref_counts[state.id] -= 1
        if not self.ref_counts[state.id]:
            del self.ref_counts[state.id]
            state.close()

    def get_data(self, state=None, request=None):
        get_data_method = (self._in_memory_get_data if self.load_in_memory
                           else self._out_of_memory_get_data)
        return get_data_method(state=state, request=request)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        return self.filter_sources([data_source[request] for data_source
                                    in self.data_sources])

    def _out_of_memory_get_data(self, state=None, request=None):
        if isinstance(request, slice):
            request = slice(request.start + self._start,
                            request.stop + self._start, request.step)
        elif isinstance(request, list):
            request = [index + self.start for index in request]
        else:
            raise ValueError
        return self.filter_sources([data_source[request] for data_source in
                                    state.values()])
