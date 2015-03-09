import h5py
import tables
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


class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * All splits reside in the same file, as subgroups of the root.
    * Data sources, such as features or targets, are children of the
      split subgroups, and their names define the source names.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    which_set : str
        Which subgroup to use.
    start : int
        Start index
    stop : int
        Stop index
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.

    """
    ref_counts = dict()

    def __init__(self, path, which_set, start=None, stop=None,
                 driver=None, **kwargs):
        self.path = path
        self.which_set = which_set
        self.driver = driver

        handle = self.open()
        self.provides_sources = handle[self.which_set].keys()
        shapes = [data_source.shape for data_source in
                  handle[self.which_set].values()]
        if any(s[0] != shapes[0][0] for s in shapes):
            raise ValueError("data sources vary in length")
        self.start = 0 if start is None else start
        self.stop = shapes[0][0] if stop is None else stop
        self.num_examples = self.stop - self.start
        self.close(handle)

        super(H5PYDataset, self).__init__(**kwargs)

    def _get_file_id(self):
        try:
            return filter(
                lambda x: x.name == self.path, self.ref_counts.keys()).next()
        except StopIteration:
            return self.path

    def open(self):
        file_id = self._get_file_id()
        state = h5py.File(name=file_id, mode="r", driver=self.driver)
        self.ref_counts[state.id] = self.ref_counts.get(state.id, 0) + 1
        return state

    def close(self, state):
        self.ref_counts[state.id] -= 1
        if not self.ref_counts[state.id]:
            del self.ref_counts[state.id]
            state.close()

    def get_data(self, state=None, request=None):
        if isinstance(request, slice):
            request = slice(request.start + self.start,
                            request.stop + self.start, request.step)
        elif isinstance(request, list):
            request = [index + self.start for index in request]
        else:
            raise ValueError
        return self.filter_sources([data_source[request] for data_source in
                                    state[self.which_set].values()])
