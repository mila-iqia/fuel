# -*- coding: utf-8 -*-

import numpy

from collections import OrderedDict

from fuel.datasets import IndexableDataset


class Spiral(IndexableDataset):
    u"""Toy dataset containing points sampled from spirals on a 2d plane.

    The dataset contains 3 sources:

    * features -- the (x, y) position of the datapoints
    * position -- the relative position on the spiral arm
    * label -- the class labels (spiral arm)

    .. plot::

        from fuel.datasets.toy import Spiral

        ds = Spiral(classes=3)
        features, position, label = ds.get_data(None, slice(0, 500))

        plt.title("Datapoints drawn from Spiral(classes=3)")
        for l, m in enumerate(['o', '^', 'v']):
            mask = label == l
            plt.scatter(features[mask,0], features[mask,1],
                        c=position[mask], marker=m, label="label==%d"%l)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel("features[:,0]")
        plt.ylabel("features[:,1]")
        plt.show()

    Parameters
    ----------
    n_datapoints : int
        Number of datapoints to create
    classes : int
        Number of spiral arms
    cycles : float
    sd : float
        Normal distributed noise with standard deviation *ds* is added to the
        features.
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
