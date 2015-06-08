# -*- coding: utf-8 -*-
import os

from fuel import config
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX


class TorontoFaceDatabase(H5PYDataset):
    u"""Toronto Face Database.

    The Toronto Face Dataset [TFD] contains face images from 8 different
    sources subsumed into a common dataset. To homogenize the images they
    where all centered and cropped using a face detector.

    In order to use this dataset you need to get the permissions/licenses
    from all sources:

    * JACFEE   http://www.humintell.com/for-use-in-research/
               ($175  Get the Standard Expressor version)
    * POFA     http://www.paulekman.com/product/
                pictures-of-facial-affect-pofa/ ($175)
    * KDEF     http://www.emotionlab.se/resources/kdef
    * CK+      http://www.pitt.edu/~emotion/ck-spread.htm
    * MMI      http://www.mmifacedb.com/
    * MSFDE    http://psychophysiolab.com/msfde/terms.php
    * NimStim  http://www.macbrain.org/resources.htm
    * RadBound http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=register

    .. [TFD] Joshua Susskind, Adam Anderson, and Geoffrey Hinton,
       *The Toronto Face Database*,
       Technical Report, July 2010, UTML TR 2010—001.

    Parameters
    ----------
    size : int
        Either 48 or 96 which will give images of size 48x48 or 96x96.
        Defaults to 48.
    which_set :
        One of 'unlabeled', 'train', 'valid' or 'test'.
    """
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_set, size=48, **kwargs):
        if size not in (48, 96):
            raise ValueError("size argument must be 48 or 96")

        self.filename = 'toronto_face_database{}.hdf5'.format(size)

        kwargs.setdefault('load_in_memory', True)
        super(TorontoFaceDatabase, self).__init__(
            self.data_path, which_set, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
