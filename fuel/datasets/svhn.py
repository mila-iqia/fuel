# -*- coding: utf-8 -*-
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class SVHN(H5PYDataset):
    """The Street View House Numbers (SVHN) dataset.

    SVHN [SVHN] is a real-world image dataset for developing machine
    learning and object recognition algorithms with minimal requirement
    on data preprocessing and formatting. It can be seen as similar in
    flavor to MNIST [LBBH] (e.g., the images are of small cropped
    digits), but incorporates an order of magnitude more labeled data
    (over 600,000 digit images) and comes from a significantly harder,
    unsolved, real world problem (recognizing digits and numbers in
    natural scene images). SVHN is obtained from house numbers in
    Google Street View images.

    Parameters
    ----------
    which_format : {1, 2}
        SVHN format 1 contains the full numbers, whereas SVHN format 2
        contains cropped digits.
    which_sets : tuple of str
        Which split to load. Valid values are 'train', 'test' and 'extra',
        corresponding to the training set (73,257 examples), the test
        set (26,032 examples) and the extra set (531,131 examples).
        Note that SVHN does not have a validation set; usually you will
        create your own training/validation split using the `subset`
        argument.

    """
    _filename = 'svhn_format_{}.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_format, which_sets, **kwargs):
        self.which_format = which_format
        super(SVHN, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

    @property
    def filename(self):
        return self._filename.format(self.which_format)
