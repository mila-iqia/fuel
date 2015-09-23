from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class CIFAR10(H5PYDataset):
    """The CIFAR10 dataset of natural images.

    This dataset is a labeled subset of the ``80 million tiny images``
    dataset [TINY]. It consists of 60,000 32 x 32 colour images in 10
    classes, with 6,000 images per class. There are 50,000 training
    images and 10,000 test images [CIFAR10].

    .. [CIFAR10] Alex Krizhevsky, *Learning Multiple Layers of Features
       from Tiny Images*, technical report, 2009.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (50,000 examples) and the test
        set (10,000 examples). Note that CIFAR10 does not have a
        validation set; usually you will create your own
        training/validation split using the `subset` argument.

    """
    filename = 'cifar10.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(CIFAR10, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
