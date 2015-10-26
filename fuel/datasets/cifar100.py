from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class CIFAR100(H5PYDataset):
    """The CIFAR100 dataset of natural images.

    This dataset is a labeled subset of the ``80 million tiny images``
    dataset [TINY]. It consists of 60,000 32 x 32 colour images labelled
    into 100 fine-grained classes and 20 super-classes. There are
    600 images per fine-grained class. There are 50,000 training
    images and 10,000 test images [CIFAR100].

    The dataset contains three sources:
    - features: the images themselves,
    - coarse_labels: the superclasses 1-20,
    - fine_labels: the fine-grained classes 1-100.

    .. [TINY] Antonio Torralba, Rob Fergus and William T. Freeman,
       *80 million tiny images: a large dataset for non-parametric
       object and scene recognition*, Pattern Analysis and Machine
       Intelligence, IEEE Transactions on 30.11 (2008): 1958-1970.

    .. [CIFAR100] Alex Krizhevsky, *Learning Multiple Layers of Features
       from Tiny Images*, technical report, 2009.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (50,000 examples) and the test
        set (10,000 examples). Note that CIFAR100 does not have a
        validation set; usually you will create your own
        training/validation split using the `subset` argument.

    """
    filename = 'cifar100.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(CIFAR100, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
