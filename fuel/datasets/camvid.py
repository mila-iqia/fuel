from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class Camvid(H5PYDataset):
    '''The CamVid motion based segmentation dataset
    The Cambridge-driving Labeled Video Database (CamVid) [Camvid1]_ provides
    high-quality videos acquired at 30 Hz with the corresponding
    semantically labeled masks at 1 Hz and in part, 15 Hz. The ground
    truth labels associate each pixel with one of 32 semantic classes.
    This loader is intended for the SegNet version of the CamVid dataset,
    that resizes the original data to 360 by 480 resolution and remaps
    the ground truth to a subset of 11 semantic classes, plus a void
    class.
    The dataset should be downloaded from [Camvid2]_ into the
    `shared_path` (that should be specified in the config.ini according
    to the instructions in ../README.md).
    Parameters
    ----------
    which_sets: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    References
    ----------
    .. [Camvid1] http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
    .. [Camvid2]
       https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    '''

    filename = 'camvid.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features', 'labels'))

    def __init__(self, which_sets, **kwargs):
        super(Camvid, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
