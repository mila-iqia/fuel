# -*- coding: utf-8 -*-
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import rgb_images_from_encoded_bytes
from fuel.utils import find_in_data_path


class ILSVRC2010(H5PYDataset):
    u"""The ILSVRC2010 Dataset.

    The ImageNet Large-Scale Visual Recognition Challenge [ILSVRC]
    is an annual computer vision competition testing object classification
    and detection at large-scale. This is a wrapper around the data for
    the 2010 competition, which is (as of 2015) the only year for which
    test data groundtruth is available.

    Note that the download site for the images is not publicly
    accessible. To download the images, you may sign up for an account
    at [SIGNUP].

    .. [ILSVRC] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause,
       Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya
       Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
       *ImageNet Large Scale Visual Recognition Challenge*. IJCV, 2015.

    .. [SIGNUP] http://www.image-net.org/signup

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' (1.2M examples)
        'valid' (150,000 examples), and 'test' (50,000 examples).

    """
    filename = 'ilsvrc2010.hdf5'
    default_transformers = rgb_images_from_encoded_bytes(('encoded_images',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(ILSVRC2010, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


class ILSVRC2012(H5PYDataset):
    u"""The ILSVRC2012 Dataset.

    The ImageNet Large-Scale Visual Recognition Challenge [ILSVRC]
    is an annual computer vision competition testing object classification
    and detection at large-scale. This is a wrapper around the data for
    the 2012 competition.

    Note that the download site for the images is not publicly
    accessible. To downlaod the images, you may sign up for an account
    at [SIGNUP].

    .. [ILSVRC] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause,
       Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya
       Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
       *ImageNet Large Scale Visual Recognition Challenge*. IJCV, 2015.

    .. [SIGNUP] http://www.image-net.org/signup

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' (1,281,167 examples)
        'valid' (50,000 examples), and 'test' (100,000 examples).

    """
    filename = 'ilsvrc2012.hdf5'
    default_transformers = rgb_images_from_encoded_bytes(('encoded_images',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(ILSVRC2012, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
