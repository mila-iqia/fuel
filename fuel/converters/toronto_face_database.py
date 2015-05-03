import os

import h5py
import numpy

from scipy.io import loadmat

from fuel.converters.base import fill_hdf5_file


def toronto_face_database(input_directory, save_path):
"""Converts the Toronto Face Database to HDF5.

    ToDo

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
