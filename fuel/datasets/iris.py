from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path


class Iris(H5PYDataset):
    u"""Iris dataset.

    Iris [IRIS] is a simple pattern recognition dataset, which consist of
    3 classes of 50 examples each having 4 real-valued features each, where
    each class refers to a type of iris plant. It is accessible through the
    UCI Machine Learning repository [UCIIRIS].

    .. [IRIS] Ronald A. Fisher, *The use of multiple measurements in
       taxonomic problems*, Annual Eugenics, 7, Part II, 179-188,
       September 1936.
    .. [UCIIRIS] https://archive.ics.uci.edu/ml/datasets/Iris

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid value is 'all'
        corresponding to 150 examples.

    """
    filename = 'iris.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(Iris, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
