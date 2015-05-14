from itertools import product
from collections import defaultdict, OrderedDict

import h5py
import numpy
import tables
from six.moves import zip

from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('nodes', 'h5file')
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
        self.h5file = tables.open_file(path, mode="r")
        node = self.h5file.getNode('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]

    def load(self):
        self.open_file(self.path)

    def close(self):
        self.h5file.close()

    def get_data(self, state=None, request=None):
        """ Returns data from HDF5 dataset.

        .. note:: The best performance if `request` is a slice.

        """
        if self.start:
            if isinstance(request, slice):
                request = slice(request.start + self.start,
                                request.stop + self.start, request.step)
                data = [node[request] for node in self.nodes]
            elif isinstance(request, list):
                request = [index + self.start for index in request]
                data = [node[request, ...] for node in self.nodes]
            else:
                raise ValueError
        return data


@do_not_pickle_attributes('data_sources', 'external_file_handle')
class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * Data sources are not explicitly split. Instead, splits are defined
      in the `split` attribute of the root group. It's expected to be a
      1D numpy array of compound ``dtype`` with six fields, organized as
      follows:

      1. ``split`` : string identifier for the split name
      2. ``source`` : string identifier for the source name
      3. ``start`` : start index (inclusive) of the split in the source
         array
      4. ``stop`` : stop index (exclusive) of the split in the source
         array
      5. ``available`` : boolean, ``False`` is this split is not available
         for this source
      6. ``comment`` : comment string

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        HDF5 file handle, or path to the HDF5 file.
    which_set : str
        Which split to use.
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
    sort_indices : bool, optional
        Whether to explicitly sort requested indices when data is
        requested in the form of a list of indices. Defaults to `True`.
        This flag can be set to `False` for greater performance. In
        that case, it is the user's responsibility to make sure that
        indices are ordered.

    """
    interface_version = '0.1'
    _ref_counts = defaultdict(int)
    _file_handles = {}

    def __init__(self, file_or_path, which_set, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        if isinstance(file_or_path, h5py.File):
            self.path = file_or_path.filename
            self.external_file_handle = file_or_path
        else:
            self.path = file_or_path
            self.external_file_handle = None
        self.driver = driver
        self.sort_indices = sort_indices
        if which_set not in self.available_splits:
            raise ValueError(
                "'{}' split is not provided by this ".format(which_set) +
                "dataset. Available splits are " +
                "{}.".format(self.available_splits))
        self.which_set = which_set
        subset = subset if subset else slice(None)
        if subset.step not in (1, None):
            raise ValueError("subset.step must be either 1 or None")
        self._subset_template = subset
        self.load_in_memory = load_in_memory

        kwargs.setdefault('axis_labels', self.load_axis_labels())
        super(H5PYDataset, self).__init__(**kwargs)

    @staticmethod
    def create_split_array(split_dict):
        """Create a valid array for the `split` attribute of the root node.

        Parameters
        ----------
        split_dict : dict
            Maps split names to dict. Those dict map source names to
            tuples. Those tuples contain two or three elements:
            the start index, the stop index and (optionally) a comment.
            If a particular split/source combination isn't present
            in the split dict, it's considered as unavailable and the
            `available` element will be set to `False` it its split array
            entry.

        """
        # Determine maximum split, source and string lengths
        split_len = max(len(split) for split in split_dict)
        sources = set()
        comment_len = 1
        for split in split_dict.values():
            sources |= set(split.keys())
            for val in split.values():
                if len(val) == 3:
                    comment_len = max([comment_len, len(val[-1])])
        sources = sorted(list(sources))
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(split_dict) * len(sources),
            dtype=numpy.dtype([
                ('split', 'a', split_len),
                ('source', 'a', source_len),
                ('start', numpy.int64, 1), ('stop', numpy.int64, 1),
                ('available', numpy.bool, 1),
                ('comment', 'a', comment_len)]))

        # Fill split array
        for i, (split, source) in enumerate(product(split_dict, sources)):
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
            # Workaround for H5PY being unable to store unicode type
            split_array[i]['split'] = split.encode('utf8')
            split_array[i]['source'] = source.encode('utf8')
            split_array[i]['start'] = start
            split_array[i]['stop'] = stop
            split_array[i]['available'] = available
            split_array[i]['comment'] = comment.encode('utf8')

        return split_array

    @staticmethod
    def parse_split_array(split_array):
        split_dict = OrderedDict()
        for row in split_array:
            split, source, start, stop, available, comment = row
            split = split.decode('utf8')
            source = source.decode('utf8')
            comment = comment.decode('utf8')
            if available:
                if split not in split_dict:
                    split_dict[split] = OrderedDict()
                split_dict[split][source] = (start, stop, comment)
        return split_dict

    @property
    def split_dict(self):
        if not hasattr(self, '_split_dict'):
            self._out_of_memory_open()
            handle = self._file_handle
            split_array = handle.attrs['split']
            self._split_dict = H5PYDataset.parse_split_array(split_array)
            self._out_of_memory_close()
        return self._split_dict

    def load_axis_labels(self):
        self._out_of_memory_open()
        handle = self._file_handle
        axis_labels = {}
        for source_name in handle:
            axis_labels[source_name] = tuple(
                dim.label for dim in handle[source_name].dims)
        self._out_of_memory_close()
        return axis_labels

    @property
    def available_splits(self):
        return tuple(self.split_dict.keys())

    @property
    def provides_sources(self):
        return tuple(self.split_dict[self.which_set].keys())

    @property
    def subsets(self):
        if not hasattr(self, '_subsets'):
            subsets = [self._subset_template for source in self.sources]
            num_examples = None
            for i, source_name in enumerate(self.sources):
                start, stop = self.split_dict[self.which_set][source_name][:2]
                subset = subsets[i]
                subset = slice(
                    start if subset.start is None else subset.start,
                    stop if subset.stop is None else subset.stop,
                    subset.step)
                subsets[i] = subset
                if num_examples is None:
                    num_examples = subset.stop - subset.start
                if num_examples != subset.stop - subset.start:
                    raise ValueError("sources have different lengths")
            self._subsets = subsets
        return self._subsets

    def load(self):
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None
        if self.load_in_memory:
            self._out_of_memory_open()
            handle = self._file_handle
            self.data_sources = tuple(
                handle[source_name][subset] for source_name, subset in
                zip(self.sources, self.subsets))
            self._out_of_memory_close()
        else:
            self.data_sources = None

    @property
    def num_examples(self):
        return self.subsets[0].stop - self.subsets[0].start

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        if not self._external_file_handle:
            if self.path not in self._file_handles:
                handle = h5py.File(
                    name=self.path, mode="r", driver=self.driver)
                self._file_handles[self.path] = handle
            self._ref_counts[self.path] += 1

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close()

    def _out_of_memory_close(self):
        if not self._external_file_handle:
            self._ref_counts[self.path] -= 1
            if not self._ref_counts[self.path]:
                del self._ref_counts[self.path]
                self._file_handles[self.path].close()
                del self._file_handles[self.path]

    @property
    def _file_handle(self):
        if self._external_file_handle:
            return self._external_file_handle
        elif self.path in self._file_handles:
            return self._file_handles[self.path]
        else:
            raise IOError('no open handle for file {}'.format(self.path))

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            return self._in_memory_get_data(state=state, request=request)
        else:
            return self._out_of_memory_get_data(state=state, request=request)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        return tuple(data_source[request] for data_source in self.data_sources)

    def _out_of_memory_get_data(self, state=None, request=None):
        rval = []
        handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            if isinstance(request, slice):
                req = slice(request.start + subset.start,
                            request.stop + subset.start, request.step)
                data = handle[source_name][req]
            elif isinstance(request, list):
                req = [index + subset.start for index in request]
                if self.sort_indices:
                    indices = numpy.argsort(req)
                    source = handle[source_name]
                    data = numpy.empty(
                        shape=(len(req),) + source.shape[1:],
                        dtype=source.dtype)
                    data[indices] = source[numpy.array(req)[indices], ...]
                else:
                    data = handle[source_name][req]
            else:
                raise ValueError
            rval.append(data)
        return tuple(rval)
