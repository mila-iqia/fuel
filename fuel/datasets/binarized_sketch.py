# -*- coding: utf-8 -*-
import logging
import os
from collections import OrderedDict

import numpy

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes

logger = logging.getLogger(__name__)


@do_not_pickle_attributes('indexables')
class BinarizedSketch(IndexableDataset):
    u"""
    Load the data created in the Sketch.ipynb notebook
    The data can be found at s3://udidraw/binarized_sketch.tgz
    The two npy files should be placed in the directory binarized_sketch in fuel data directory

    The original data was downloaded from
    http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
    @article{eitz2012hdhso,
        author={Eitz, Mathias and Hays, James and Alexa, Marc},
        title={How Do Humans Sketch Objects?},
        journal={ACM Trans. Graph. (Proc. SIGGRAPH)},
        year={2012},
        volume={31},
        number={4},
        pages = {44:1--44:10}
    }

    Parameters
    ----------
    which_set : 'train' or 'test'
        Whether to load the training set (18000 samples) or the test set (2000 samples).
    """
    provides_sources = ('features',)
    folder = 'binarized_sketch'
    files = {set_: 'binarized_sketch_{}.npy'.format(set_) for set_ in
             ('train', 'test')}

    def __init__(self, which_set, flatten=True, **kwargs):
        if which_set not in ('train', 'test'):
            raise ValueError("available splits are 'train' and "
                             "'test'")
        self.which_set = which_set
        self.flatten = flatten

        super(BinarizedSketch, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_binarized_shrec())), **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path, self.folder,
                            self.files[self.which_set])

    def load(self):
        indexable, = self._load_binarized_shrec()
        self.indexables = [indexable[self.start:self.stop]]

    def _load_binarized_shrec(self):
        images = numpy.load(self.data_path).astype(config.floatX)
        return [images]
