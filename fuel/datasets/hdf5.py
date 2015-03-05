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
