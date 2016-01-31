from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class CelebA(H5PYDataset):
    """The CelebFaces Attributes Dataset (CelebA) dataset.

    CelebA is a large-scale face
    attributes dataset with more than 200K celebrity images, each
    with 40 attribute annotations. The images in this dataset cover
    large pose variations and background clutter. CelebA has large
    diversities, large quantities, and rich annotations, including:

    * 10,177 number of identities
    * 202,599 number of face images
    * 5 landmark locations per image
    * 40 binary attributes annotations per image.

    The dataset can be employed as the training and test sets for
    the following computer vision tasks:

    * face attribute recognition
    * face detection
    * landmark (or facial part) localization

    Parameters
    ----------
    which_format : {'aligned_cropped, '64'}
        Either the aligned and cropped version of CelebA, or
        a 64x64 version of it.
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'valid' and
        'test' corresponding to the training set (162,770 examples), the
        validation set (19,867 examples) and the test set (19,962
        examples).

    """
    _filename = 'celeba_{}.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_format, which_sets, **kwargs):
        self.which_format = which_format
        super(CelebA, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

    @property
    def filename(self):
        return self._filename.format(self.which_format)
