from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path


class Adult(H5PYDataset):
    filename = 'adult.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(Adult, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs
        )
