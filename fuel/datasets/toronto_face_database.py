# -*- coding: utf-8 -*-
import logging
import os

from fuel import config
from fuel.datasets import H5PYDataset

logger = logging.getLogger(__name__)


class TorontoFaceDatabase(H5PYDataset):
    u"""
    
    XXX

    Parameters
    ----------
    start : int
    stop : int
    flatten : list

    """
    folder = 'faces/TFD'
    filename = 'tfd48.h5'

    def __init__(self, start=None, stop=None, load_in_memory=True, **kwargs):
        if start is not None:
            subset = slice(start, stop)
        else:
            subset = None

        super(TorontoFaceDatabase, self).__init__(
            path=self.data_path, 
            subset=subset,
            load_in_memory=load_in_memory, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.folder, self.filename)
