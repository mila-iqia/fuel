import os

import h5py
import numpy

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file


def toronto_face_database(input_directory, save_path):
    """Converts the binarized MNIST dataset to HDF5.

    Converts the binarized MNIST dataset used in R. Salakhutdinov's DBN
    paper [DBN] to an HDF5 dataset compatible with
    :class:`fuel.datasets.BinarizedMNIST`. The converted dataset is
    saved as 'binarized_mnist.hdf5'.

    This method assumes the existence of the files
    `binarized_mnist_{train,valid,test}.amat`, which are accessible
    through Hugo Larochelle's website [HUGO].

    .. [DBN] Ruslan Salakhutdinov and Iain Murray, *On the Quantitative
       Analysis of Deep Belief Networks*, Proceedings of the 25th
       international conference on Machine learning, 2008, pp. 872-879.

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    input_directory : str
        Directory in which the required input files reside.
    save_path : str
        Where to save the converted dataset.

    """
    h5file = h5py.File(save_path, mode="w")

    tfd = loadmat(os.path.join(input_directory, 'TFD_48x48.mat'))
    
    folds = tfd['folds']
    features = tfd['images'].reshape([-1, 1, 48, 48])
    labs_ex = tfd['labs_ex']
    labs_id = tfd['labs_id']

    unlabeled_mask = folds[:,0] == 0
    train1_mask = folds[:,0] == 1
    valid1_mask = folds[:,0] == 2
    test1_mask  = folds[:,0] == 3

    import ipdb; ipdb.set_trace()

    data = (('unlabeled', 'features', features[unlabeled_mask]),
            ('train', 'features', features[train1_mask]),
            ('train', 'labs_ex' , labs_ex[train1_mask]),
            ('train', 'labs_id' , labs_id[train1_mask]),
            ('valid', 'features', features[valid1_mask]),
            ('valid', 'labs_ex' , labs_ex[valid1_mask]),
            ('valid', 'labs_id' , labs_id[valid1_mask]),
            ('test', 'features', features[test1_mask]),
            ('test', 'labs_ex' , labs_ex[test1_mask]),
            ('test', 'labs_id' , labs_id[test1_mask]),
    )
    fill_hdf5_file(h5file, data)

    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label

    h5file.flush()
    h5file.close()
