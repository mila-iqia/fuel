from fuel.datasets import H5PYDataset
from fuel.transformers import ScaleAndShift
from fuel.utils import find_in_data_path


class MJSynth(H5PYDataset):
    """The MJSynth dataset of cropped words images.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'test' and 'val'.

    """
    filename = 'mjsynth.hdf5'

    default_transformers = ((ScaleAndShift, [1 / 255.0, 0],
                             {'which_sources': ('features',)}),)

    def __init__(self, which_sets, **kwargs):
        super(MJSynth, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
