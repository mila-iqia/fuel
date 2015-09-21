# -*- coding: utf-8 -*-
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class Weizmann_horse(H5PYDataset):
    u"""Weizmann horses dataset.

    The Weizmann Horse Database [ECCV2002] consists of 328 side-view color
    images of horses that were also manually segmented. The images were
    randomly collected from the import WWW to evaluate the top-down
    segmentation scheme as well as its combination with bottom-up processing.
    For further elaboration, please refer to the related publications
    (ECCV2002, ECCV2004, CVPR2004).
    It is accessible through Eran Borenstein's website [BORENSTEIN].


    .. [ECCV2002] E. Borenstein and S. Ullman,
       *Class-Specific, Top-Down Segmentation*,
       Springer-Verlag LNCS 2351,
       European Conference on Computer Vision (ECCV), May 2002

    .. [ECCV2004] E. Borenstein and S. Ullman,
       *Learning to Segment*,
       Springer-Verlag LNCS 3023;
       European Conference on Computer Vision (ECCV), May 2004

    .. [CVPR2004] E. Borenstein, E. Sharon and S. Ullman,
       *Combining Top-down and Bottom-up Segmentation*,
       Proceedings IEEE workshop on Perceptual Organization in Computer Vision,
       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June
       2004

    .. [BORENSTEIN] http://www.msri.org/people/members/eranb/

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (50,000 examples) and the test
        set (10,000 examples).

    """
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, filename=None, resize=False,
                 resize_size=-1, zero_pad=True, crop=False, split=[.44, .22],
                 **kwargs):
        kwargs.setdefault('load_in_memory', True)
        if filename is None:
            filename = 'weizmann_horse'
            if resize:
                filename += '_' + str(resize_size)
                if zero_pad:
                    filename += '_zero_padded'
                else:
                    filename += '_cropped'
            filename += '_' + str(split[0]) + '_' + str(split[1]) + '.hdf5'

        super(Weizmann_horse, self).__init__(
            file_or_path=find_in_data_path(filename), **kwargs)
