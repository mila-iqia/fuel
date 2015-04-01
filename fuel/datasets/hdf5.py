from itertools import product
from collections import defaultdict

import h5py
import numpy
import tables

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
    subset : slice, optional
        A slice of data *within the context of the split* to use. Defaults
        to `None`, in which case the whole split is used. **Note:
        at the moment, `slice.step` must be either 1 or `None`.**
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    flatten : list of str, optional
        Which sources to flatten as a 2D array, if any. Defaults to `None`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.

    """
    ref_counts = defaultdict(int)

    def __init__(self, path, which_set, subset=None, load_in_memory=False,
                 flatten=None, driver=None, **kwargs):
        self.path = path
        self.driver = driver
        if which_set not in self.available_splits:
            raise ValueError(
                "'{}' split is not provided by this dataset".format(which_set))
        self.which_set = which_set
        subset = subset if subset else slice(None, None, None)
        if subset.step not in (1, None):
            raise ValueError("subset.step must be either 1 or None")
        self.subsets = [subset for source in self.provides_sources]
        self.load_in_memory = load_in_memory
        self.flatten = [] if flatten is None else flatten
        for source in self.flatten:
            if source not in self.provides_sources:
                raise ValueError(
                    "trying to flatten source '{}' which is ".format(source) +
                    "not provided by the '{}' split".format(self.which_set))

        super(H5PYDataset, self).__init__(**kwargs)

        self.load()

    @staticmethod
    def create_split_array(split_dict):
        # Determine maximum split, source and string lengths
        splits = tuple(split_dict.keys())
        split_len = max(len(split) for split in splits)
        sources = set()
        comment_len = 1
        for split in split_dict.values():
            sources |= set(split.keys())
            for val in split.values():
                if len(val) == 3:
                    comment_len = max([comment_len, len(val[-1])])
        sources = tuple(sources)
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(splits) * len(sources),
            dtype=numpy.dtype([
                ('split', numpy.str_, split_len),
                ('source', numpy.str_, source_len),
                ('start', numpy.int64, 1), ('stop', numpy.int64, 1),
                ('available', numpy.bool, 1),
                ('comment', numpy.str_, comment_len)]))

        # Fill split array
        for i, (split, source) in enumerate(product(splits, sources)):
            if source in split_dict[split]:
                start, stop = split_dict[split][source][:2]
                available = True
                # Workaround for bug when pickling an empty string
                comment = '.'
                if len(split_dict[split][source]) == 3:
                    comment = split_dict[split][source][2]
                    if not comment:
                        comment = '.'
            else:
                (start, stop, available, comment) = (0, 0, False, '.')
            split_array[i]['split'] = split
            split_array[i]['source'] = source
            split_array[i]['start'] = start
            split_array[i]['stop'] = stop
            split_array[i]['available'] = available
            split_array[i]['comment'] = comment

        return split_array

    @staticmethod
    def parse_split_array(split_array):
        split_dict = defaultdict(dict)
        for row in split_array:
            split, source, start, stop, available, comment = row
            if available:
                split_dict[split][source] = (start, stop, comment)
        return dict(split_dict)

    def _get_file_id(self):
        file_id = [f for f in self.ref_counts.keys() if f.name == self.path]
        if not file_id:
            return self.path
        file_id, = file_id
        return file_id

    @property
    def split_dict(self):
        if not hasattr(self, '_split_dict'):
            handle = self._out_of_memory_open()
            split_array = handle.attrs['split']
            self._split_dict = H5PYDataset.parse_split_array(split_array)
            self._out_of_memory_close(handle)
        return self._split_dict

    @property
    def available_splits(self):
        return tuple(self.split_dict.keys())

    @property
    def provides_sources(self):
        return tuple(self.split_dict[self.which_set].keys())

    def load(self):
        handle = self._out_of_memory_open()
        num_examples = None
        for i, (source_name, data_source) in enumerate(handle.items()):
            if source_name in self.provides_sources:
                start, stop = self.split_dict[self.which_set][source_name][:2]
                subset = self.subsets[i]
                subset = slice(
                    start if subset.start is None else subset.start,
                    stop if subset.stop is None else subset.stop,
                    subset.step)
                self.subsets[i] = subset
                if num_examples is None:
                    num_examples = subset.stop - subset.start
                if num_examples != subset.stop - subset.start:
                    raise ValueError("sources have different lengths")
        self.num_examples = num_examples
        if self.load_in_memory:
            data_sources = []
            for i, (source_name, data_source) in enumerate(handle.items()):
                if source_name in self.sources:
                    data = data_source[self.subsets[i]]
                    if source_name in self.flatten:
                        data = data.reshape((data.shape[0], -1))
                    data_sources.append(data)
            self.data_sources = data_sources
        else:
            self.data_sources = None
        self._out_of_memory_close(handle)

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        file_id = self._get_file_id()
        state = h5py.File(name=file_id, mode="r", driver=self.driver)
        self.ref_counts[state.id] += 1
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
        if self.load_in_memory:
            return self._in_memory_get_data(state=state, request=request)
        else:
            return self._out_of_memory_get_data(state=state, request=request)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        return self.filter_sources([data_source[request] for data_source
                                    in self.data_sources])

    def _out_of_memory_get_data(self, state=None, request=None):
        rval = []
        for i, (source_name, data_source) in enumerate(state.items()):
            if source_name not in self.sources:
                continue
            subset = self.subsets[i]
            if isinstance(request, slice):
                request = slice(request.start + subset.start,
                                request.stop + subset.start, request.step)
            elif isinstance(request, list):
                request = [index + subset.start for index in request]
            else:
                raise ValueError
            data = data_source[request]
            if source_name in self.flatten:
                data = data.reshape((data.shape[0], -1))
            rval.append(data)
        return tuple(rval)
