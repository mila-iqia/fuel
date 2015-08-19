from itertools import product
from collections import defaultdict

import h5py
import numpy
import six
import tables
from six.moves import zip, range

from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes, iterable_fancy_indexing


@do_not_pickle_attributes('nodes', 'h5file')
class PytablesDataset(Dataset):
    """A pytables dataset.

    An HDF5 Dataset which was created with pytables. The dataset should
    have the following structure: `/<data_node>/paths/to/sources`. In
    order to have train/validation/test split you may want to open
    several datasets with different data nodes or source paths. It is
    also possible to use start and stop arguments to split your dataset.

    Parameters
    ----------
    sources : tuple of strings
        Sources which the dataset returns.
    start : int
        Start index. Optional, by default is 0.
    stop : int
        Stop index. Optional, if is not provided, will be set to the
        number of rows of the first source.
    data_node : str
        Parent data node in HDF5 file, all path are relative to this node.
    sources_in_file : tuple of strings
        Names of nodes in HDF5 file which contain sources. Should the same
        length as `sources`.
        Optional, if not set will be equal to `sources`.

    """
    def __init__(self, path, sources, start=0, stop=None, data_node='Data',
                 sources_in_file=None):
        if sources_in_file is None:
            sources_in_file = sources
        self.sources_in_file = sources_in_file
        self.provides_sources = sources
        self.path = path
        self.data_node = data_node
        self.start = start
        self.stop = stop
        self.nodes = None
        self.open_file(path)
        super(PytablesDataset, self).__init__(self.provides_sources)

    def open_file(self, path):
        self.h5file = tables.open_file(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start

    def load(self):
        self.open_file(self.path)

    def close_file(self):
        self.h5file.close()
        del self._h5file
        del self._nodes

    def get_data(self, state=None, request=None):
        """ Returns data from HDF5 dataset.

        .. note:: The best performance if `request` is a slice.

        """
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


@do_not_pickle_attributes('data_sources', 'external_file_handle',
                          'source_shapes', 'subsets')
class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * Data sources are not explicitly split. Instead, splits are defined
      in the `split` attribute of the root group. It's expected to be a
      1D numpy array of compound ``dtype`` with seven fields, organized as
      follows:

      1. ``split`` : string identifier for the split name
      2. ``source`` : string identifier for the source name
      3. ``start`` : start index (inclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      4. ``stop`` : stop index (exclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      5. ``indices`` : h5py.Reference, reference to a dataset containing
         subset indices for this split/source pair. If it's a null
         reference, ``start`` and ``stop`` are used.
      6. ``available`` : boolean, ``False`` is this split is not available
         for this source
      7. ``comment`` : comment string

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        HDF5 file handle, or path to the HDF5 file.
    which_sets : iterable of str
        Which split(s) to use. If one than more split is requested,
        the provided sources will be the intersection of provided
        sources for these splits. **Note: for all splits that are
        specified as a list of indices, those indices will get sorted
        no matter what.**
    subset : {slice, list of int}, optional
        Which subset of data to use *within the context of the split*.
        Can be either a slice or a list of indices. Defaults to `None`,
        in which case the whole split is used.
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.
    sort_indices : bool, optional
        HDF5 doesn't support fancy indexing with an unsorted list of
        indices. In order to allow that, the dataset can sort the list
        of indices, access the data in sorted order and shuffle back
        the data in the unsorted order. Setting this flag to `True`
        (the default) will activate this behaviour. For greater
        performance, set this flag to `False`. Note that in that case,
        it is the user's responsibility to make sure that indices are
        ordered.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide for the requested
        split.
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    vlen_sources : tuple of strings
        All sources provided by this dataset which have variable length.
    default_axis_labels : dict mapping string to tuple of strings
        Maps all sources provided by this dataset to their axis labels.

    """
    interface_version = '0.3'
    _ref_counts = defaultdict(int)
    _file_handles = {}

    def __init__(self, file_or_path, which_sets, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        if isinstance(file_or_path, h5py.File):
            self.path = file_or_path.filename
            self.external_file_handle = file_or_path
        else:
            self.path = file_or_path
            self.external_file_handle = None
        which_sets_invalid_value = (
            isinstance(which_sets, six.string_types) or
            not all(isinstance(s, six.string_types) for s in which_sets))
        if which_sets_invalid_value:
            raise ValueError('`which_sets` should be an iterable of strings')
        self.which_sets = which_sets
        self._subset_template = subset if subset else slice(None)
        self.load_in_memory = load_in_memory
        self.driver = driver
        self.sort_indices = sort_indices

        self._parse_dataset_info()

        kwargs.setdefault('axis_labels', self.default_axis_labels)
        super(H5PYDataset, self).__init__(**kwargs)

    def _parse_dataset_info(self):
        """Parses information related to the HDF5 interface.

        In addition to verifying that the `self.which_sets` split is
        available, this method sets the following attributes:

        * `provides_sources`
        * `vlen_sources`
        * `default_axis_labels`

        """
        self._out_of_memory_open()
        handle = self._file_handle
        available_splits = self.get_all_splits(handle)
        which_sets = self.which_sets
        provides_sources = None
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this ".format(split) +
                    "dataset. Available splits are " +
                    "{}.".format(available_splits))
            split_provides_sources = set(
                self.get_provided_sources(handle, split))
            if provides_sources:
                provides_sources &= split_provides_sources
            else:
                provides_sources = split_provides_sources
        self.provides_sources = tuple(sorted(provides_sources))
        self.vlen_sources = self.get_vlen_sources(handle)
        self.default_axis_labels = self.get_axis_labels(handle)
        self._out_of_memory_close()

    @staticmethod
    def create_split_array(split_dict):
        """Create a valid array for the `split` attribute of the root node.

        Parameters
        ----------
        split_dict : dict
            Maps split names to dict. Those dict map source names to
            tuples. Those tuples contain two, three or four elements:
            the start index, the stop index, (optionally) subset
            indices and (optionally) a comment.  If a particular
            split/source combination isn't present in the split dict,
            it's considered as unavailable and the `available` element
            will be set to `False` it its split array entry.

        """
        # Determine maximum split, source and string lengths
        split_len = max(len(split) for split in split_dict)
        sources = set()
        comment_len = 1
        for split in split_dict.values():
            sources |= set(split.keys())
            for val in split.values():
                if len(val) == 4:
                    comment_len = max([comment_len, len(val[-1])])
        sources = sorted(list(sources))
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(split_dict) * len(sources),
            dtype=numpy.dtype([
                ('split', 'a', split_len),
                ('source', 'a', source_len),
                ('start', numpy.int64, 1),
                ('stop', numpy.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', numpy.bool, 1),
                ('comment', 'a', comment_len)]))

        # Fill split array
        for i, (split, source) in enumerate(product(split_dict, sources)):
            if source in split_dict[split]:
                start, stop = split_dict[split][source][:2]
                available = True
                indices = h5py.Reference()
                # Workaround for bug when pickling an empty string
                comment = '.'
                if len(split_dict[split][source]) > 2:
                    indices = split_dict[split][source][2]
                if len(split_dict[split][source]) > 3:
                    comment = split_dict[split][source][3]
                    if not comment:
                        comment = '.'
            else:
                (start, stop, indices, available, comment) = (
                    0, 0, h5py.Reference(), False, '.')
            # Workaround for H5PY being unable to store unicode type
            split_array[i]['split'] = split.encode('utf8')
            split_array[i]['source'] = source.encode('utf8')
            split_array[i]['start'] = start
            split_array[i]['stop'] = stop
            split_array[i]['indices'] = indices
            split_array[i]['available'] = available
            split_array[i]['comment'] = comment.encode('utf8')

        return split_array

    @staticmethod
    def get_all_splits(h5file):
        """Returns the names of all splits of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        available_splits : tuple of str
            Names of all splits in ``h5file``.

        """
        available_splits = tuple(
            set(row['split'].decode('utf8') for row in h5file.attrs['split']))
        return available_splits

    @staticmethod
    def get_all_sources(h5file):
        """Returns the names of all sources of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        all_sources : tuple of str
            Names of all sources in ``h5file``.

        """
        all_sources = tuple(
            set(row['source'].decode('utf8') for row in h5file.attrs['split']))
        return all_sources

    @staticmethod
    def get_provided_sources(h5file, split):
        """Returns the sources provided by a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        provided_sources : tuple of str
            Names of sources provided by ``split`` in ``h5file``.

        """
        provided_sources = tuple(
            row['source'].decode('utf8') for row in h5file.attrs['split']
            if row['split'].decode('utf8') == split and row['available'])
        return provided_sources

    @staticmethod
    def get_vlen_sources(h5file):
        """Returns the names of variable-length sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        vlen_sources : tuple of str
            Names of all variable-length sources in ``h5file``.

        """
        vlen_sources = []
        for source_name in H5PYDataset.get_all_sources(h5file):
            source = h5file[source_name]
            if len(source.dims) > 0 and 'shapes' in source.dims[0]:
                if len(source.dims) > 1:
                    raise ValueError('Variable-length sources must have only '
                                     'one dimension.')
                vlen_sources.append(source_name)
        return vlen_sources

    @staticmethod
    def get_axis_labels(h5file):
        """Returns axis labels for all sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        axis_labels : dict
            Maps source names to a tuple of str representing the axis
            labels.

        """
        axis_labels = {}
        vlen_sources = H5PYDataset.get_vlen_sources(h5file)
        for source_name in H5PYDataset.get_all_sources(h5file):
            if source_name in vlen_sources:
                axis_labels[source_name] = (
                    (h5file[source_name].dims[0].label,) +
                    tuple(label.decode('utf8') for label in
                          h5file[source_name].dims[0]['shape_labels']))
            else:
                axis_labels[source_name] = tuple(
                    dim.label for dim in h5file[source_name].dims)
        return axis_labels

    @staticmethod
    def get_start_stop(h5file, split):
        """Returns start and stop indices for sources of a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        start_stop : dict
            Maps source names to a tuple of ``(start, stop)`` indices.

        """
        start_stop = {}
        for row in h5file.attrs['split']:
            if row['split'].decode('utf8') == split:
                source = row['source'].decode('utf8')
                start_stop[source] = (row['start'], row['stop'])
        return start_stop

    @staticmethod
    def get_indices(h5file, split):
        """Returns subset indices for sources of a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        indices : dict
            Maps source names to a list of indices. Note that only
            sources for which indices are specified appear in this dict.

        """
        indices = {}
        for row in h5file.attrs['split']:
            if row['split'].decode('utf8') == split:
                source = row['source'].decode('utf8')
                if row['indices']:
                    indices[source] = h5file[row['indices']]
        return indices

    @staticmethod
    def unsorted_fancy_index(request, indexable):
        """Safe unsorted list indexing.

        Some objects, such as h5py datasets, only support list indexing
        if the list is sorted.

        This static method adds support for unsorted list indexing by
        sorting the requested indices, accessing the corresponding
        elements and re-shuffling the result.

        Parameters
        ----------
        request : list of int
            Unsorted list of example indices.
        indexable : any fancy-indexable object
            Indexable we'd like to do unsorted fancy indexing on.

        """
        if len(request) > 1:
            indices = numpy.argsort(request)
            data = numpy.empty(shape=(len(request),) + indexable.shape[1:],
                               dtype=indexable.dtype)
            data[indices] = indexable[numpy.array(request)[indices], ...]
        else:
            data = indexable[request]
        return data

    def load(self):
        # If the dataset is unpickled, it makes no sense to have an external
        # file handle. However, since `load` is also called during the lifetime
        # of a dataset (e.g. if load_in_memory = True), we don't want to
        # accidentally overwrite the reference to a potential external file
        # handle, hence this check.
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None

        self._out_of_memory_open()
        handle = self._file_handle

        # Load subset slices / indices
        subsets = []
        if len(self.which_sets) > 1:
            indices = defaultdict(list)
            for split in self.which_sets:
                split_start_stop = self.get_start_stop(handle, split)
                split_indices = self.get_indices(handle, split)
                for source_name in self.sources:
                    if source_name in split_indices:
                        ind = split_indices[source_name]
                    else:
                        ind = list(range(*split_start_stop[source_name]))
                    indices[source_name] = sorted(
                        set(indices[source_name] + ind))
        else:
            start_stop = self.get_start_stop(handle, self.which_sets[0])
            indices = self.get_indices(handle, self.which_sets[0])
        num_examples = None
        for source_name in self.sources:
            subset = self._subset_template
            # If subset has a step greater than 1, we convert it to a list,
            # otherwise we won't be able to take that subset within the context
            # of a split defined by a slice.
            if hasattr(subset, 'step') and subset.step not in (1, None):
                subset = list(range(len(handle[source_name])))[subset]
            self._subset_template = subset
            if source_name in indices:
                source_subset = indices[source_name]
            else:
                source_subset = slice(*start_stop[source_name])
            if hasattr(subset, 'step') and hasattr(source_subset, 'step'):
                subset = slice(
                    source_subset.start
                    if subset.start is None else subset.start,
                    source_subset.stop
                    if subset.stop is None else subset.stop,
                    subset.step)
                subsets.append(subset)
                subset_num_examples = subset.stop - subset.start
            else:
                if hasattr(source_subset, 'step'):
                    source_subset = numpy.arange(
                        source_subset.start, source_subset.stop)
                subset = source_subset[subset]
                subsets.append(subset)
                subset_num_examples = len(subset)
            if num_examples is None:
                num_examples = subset_num_examples
            if num_examples != subset_num_examples:
                raise ValueError("sources have different lengths")
        self.subsets = subsets

        # Load data sources and source shapes
        if self.load_in_memory:
            data_sources = []
            source_shapes = []
            for source_name, subset in zip(self.sources, self.subsets):
                if hasattr(subset, 'step'):
                    data_source = handle[source_name][subset]
                else:
                    data_source = handle[source_name][list(subset)]
                data_sources.append(data_source)
                if source_name in self.vlen_sources:
                    shapes = handle[source_name].dims[0]['shapes'][subset]
                else:
                    shapes = None
                source_shapes.append(shapes)
            self.data_sources = tuple(data_sources)
            self.source_shapes = tuple(source_shapes)
        else:
            self.data_sources = None
            self.source_shapes = None

        self._out_of_memory_close()

    @property
    def num_examples(self):
        if hasattr(self.subsets[0], 'step'):
            return self.subsets[0].stop - self.subsets[0].start
        else:
            return len(self.subsets[0])

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        if not self.external_file_handle:
            if self.path not in self._file_handles:
                handle = h5py.File(
                    name=self.path, mode="r", driver=self.driver)
                self._file_handles[self.path] = handle
            self._ref_counts[self.path] += 1

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close()

    def _out_of_memory_close(self):
        if not self.external_file_handle:
            self._ref_counts[self.path] -= 1
            if not self._ref_counts[self.path]:
                del self._ref_counts[self.path]
                self._file_handles[self.path].close()
                del self._file_handles[self.path]

    @property
    def _file_handle(self):
        if self.external_file_handle:
            return self.external_file_handle
        elif self.path in self._file_handles:
            return self._file_handles[self.path]
        else:
            raise IOError('no open handle for file {}'.format(self.path))

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                for j in range(len(data[i])):
                    data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        data = [data_source[request] for data_source in self.data_sources]
        shapes = [shape[request] if shape is not None else None
                  for shape in self.source_shapes]
        return data, shapes

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (slice, list)):
            raise ValueError()
        data = []
        shapes = []
        handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            if hasattr(subset, 'step'):
                if hasattr(request, 'step'):
                    req = slice(request.start + subset.start,
                                request.stop + subset.start, request.step)
                else:
                    req = [index + subset.start for index in request]
            else:
                req = iterable_fancy_indexing(subset, request)
            if hasattr(req, 'step'):
                val = handle[source_name][req]
                if source_name in self.vlen_sources:
                    shape = handle[source_name].dims[0]['shapes'][req]
                else:
                    shape = None
            else:
                if self.sort_indices:
                    val = self.unsorted_fancy_index(req, handle[source_name])
                    if source_name in self.vlen_sources:
                        shape = self.unsorted_fancy_index(
                            req, handle[source_name].dims[0]['shapes'])
                    else:
                        shape = None
                else:
                    val = handle[source_name][req]
                    if source_name in self.vlen_sources:
                        shape = handle[source_name].dims[0]['shapes'][req]
                    else:
                        shape = None
            data.append(val)
            shapes.append(shape)
        return data, shapes
