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
    start : int
    stop : int
    flatten : list

    """
    filename = 'toronto_face_database.hdf5'

    def __init__(self, which_set, load_in_memory=True, **kwargs):
        super(TorontoFaceDatabase, self).__init__(
            path=self.data_path, 
            which_set=which_set,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.filename)
