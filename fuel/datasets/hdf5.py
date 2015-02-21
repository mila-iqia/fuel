import tables

from fuel.datasets import Dataset


class Hdf5Dataset(Dataset):
    """An HDF5 dataset

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
        self.nodes = self._open_file()
        super(Hdf5Dataset, self).__init__(self.provides_sources)

    def _open_file(self):
        h5file = tables.openFile(self.path, mode="r")
        node = h5file.getNode('/', self.data_node)

        return [getattr(node, source) for source in self.sources_in_file]

    def get_data(self, state=None, request=None):
        """ Returns data from HDF5 dataset.

        .. note:: The best performance if `request` is a slice.
        """
        data = [node[request] for node in self.nodes]
        return data

    def __getstate__(self):
        fields = self.__dict__
        # Do not return `nodes` because they are not pickable
        del fields['nodes']
        return fields

    def __setstate__(self, state):
        self.__dict__ = state
        # Open HDF5 file again and sets up `nodes`.
        self.nodes = self._open_file()

