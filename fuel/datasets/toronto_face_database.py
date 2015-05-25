# -*- coding: utf-8 -*-
import logging
import os

from fuel import config
from fuel.datasets import H5PYDataset

logger = logging.getLogger(__name__)


class TorontoFaceDatabase(H5PYDataset):
    u"""
    
    Parameters
    ----------
    which_set : 

    """
    filename = 'toronto_face_database.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(TorontoFaceDatabase, self).__init__(
            path=self.data_path, 
            which_set=which_set,
            **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
