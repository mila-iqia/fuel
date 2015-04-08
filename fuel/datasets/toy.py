# -*- coding: utf-8 -*-

import numpy

from collections import OrderedDict

from fuel.datasets import IndexableDataset


class Spiral(IndexableDataset):
    u"""Simple toy dataset containing datapoints from a spiral on a 2d plane.

    .. plot::

        from fuel.datasets.toy import Spiral

        ds = Spiral(classes=3)
        features, position, label = ds.get_data(None, slice(0, 500))

        plt.title("Datapoints drawn from Spiral(classes=3)")
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
    def __init__(self, n_datapoints=1000, classes=1, cycles=1., sd=0.0,
                 **kwargs):
        # Create dataset
        pos = numpy.random.uniform(size=n_datapoints, low=0, high=cycles)
        label = numpy.random.randint(size=n_datapoints, low=0, high=classes)

        radius = (2*pos+1) / 3.
        phase_offset = label * (2*numpy.pi) / classes

        features = numpy.zeros(shape=(n_datapoints, 2), dtype='float32')

        features[:, 0] = radius * numpy.sin(2*numpy.pi*pos + phase_offset)
        features[:, 1] = radius * numpy.cos(2*numpy.pi*pos + phase_offset)

        data = OrderedDict([
            ('features', features),
            ('position', pos),
            ('label', label),
        ])

        super(Spiral, self).__init__(data, **kwargs)
