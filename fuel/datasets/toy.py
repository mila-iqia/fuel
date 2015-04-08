# -*- coding: utf-8 -*-

import numpy 

from collections import OrderedDict

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes


class Spiral(IndexableDataset):
    u"""Simple toy dataset containing datapoints from a spiral on a 2d plane.

    .. plot::

        from fuel.datasets.toy import Spiral

        ds = Spiral()
        features, position = ds.get_data(None, slice(0, 100))

        plt.title("Datapoints drawn from Spiral(classes=2)")
        plt.scatter(features[:,0], features[:,1], c=position)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.show()

    Parameters
    ----------
    n_datapoints : int
        Number of datapoints to create
    cycles : float
    sd : float
    """
    def __init__(self, n_datapoints=1000, cycles=1., sd=0.0, **kwargs):
        # Create dataset
        pos = numpy.random.uniform(size=(n_datapoints,), low=0, high=cycles)
        radius = (2*pos+1) / 3.
        
        features = numpy.zeros(shape=(n_datapoints, 2), dtype='float32')

        features[:,0] = radius * numpy.sin(2*numpy.pi*pos)
        features[:,1] = radius * numpy.cos(2*numpy.pi*pos)

        data = OrderedDict([
            ('features', features),
            ('position', pos),
        ])

        super(Spiral, self).__init__(data, **kwargs)
