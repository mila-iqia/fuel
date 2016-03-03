from fuel.datasets import H5PYDataset
from fuel.transformers import ScaleAndShift
from fuel.utils import find_in_data_path


class DogsVsCats(H5PYDataset):
    """The Kaggle Dogs vs. Cats dataset of cats and dogs images.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test'.
        The test set is the one released on Kaggle.

    Notes
    -----
    The Dogs vs. Cats dataset does not provide an official
    validation split. Users need to create their own
    training / validation split using the `subset` argument.

    """
    filename = 'dogs_vs_cats.hdf5'

    default_transformers = ((ScaleAndShift, [1 / 255.0, 0],
                             {'which_sources': ('image_features',)}),)

    def __init__(self, which_sets, **kwargs):
        super(DogsVsCats, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
