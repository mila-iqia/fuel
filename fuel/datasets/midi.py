import os
import numpy as np
from collections import OrderedDict

from fuel.utils import do_not_pickle_attributes
from fuel.datasets import IndexableDataset
from fuel import config

@do_not_pickle_attributes('indexables')
class MidiSequence(IndexableDataset):
    def __init__(self, which_dataset, which_set='train', **kwargs):
        '''Midi Datasets

        Parameters
        ---------

        which_datset: str
           one of the following 'jsb', 'midi', 'nottingham', 'muse'
        which_set: str
           which datset split 'train', 'valid' or 'test'

        ref: Boulanger-Lewandowski et. al. Modeling Temporal Dependencies
             in High-Dimensional Sequences: Application to Polyphonic
             Music Generation and Transcription.
             Download the dataset pickled datasets from here:
             http://www-etud.iro.umontreal.ca/~boulanni/icml2012
        '''
        self.which_set = which_set
        self.which_dataset = which_dataset

        self.sources = ('features', 'targets')

        super(MidiSequence, self).__init__(
                OrderedDict(zip(self.sources,
                                self._load_data(which_dataset, which_set))),
                **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                            in zip(self.provide_sources,
                            self._load_data(
                                self.which_dataset,
                                self.which_set))
                            ]

    def _load_data(self, which_dataset, which_set):
        """
        which_dataset : choose between 'short' and 'long'
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                        "Valid values are ['train', 'valid', 'test'].")
        # Check which_dataset
        if which_dataset not in ['midi', 'nottingham', 'muse', 'jsb']:
            raise ValueError(which_set + " is not a recognized value. " +
                "Valid values are ['midi', 'nottingham', 'muse', 'jsb'].")
        _data_path = os.path.join(config.data_path, 'midi')
        if which_dataset == 'midi':
            _path = os.path.join(_data_path, "Piano-midi.de.pickle")
            max_label= 108
        elif which_dataset == 'nottingham':
            _path = os.path.join(_data_path, "Nottingham.pickle")
            max_label = 93
        elif which_dataset == 'muse':
            _path = os.path.join(_data_path, "MuseData.pickle")
            max_label = 105
        elif which_dataset == 'jsb':
            _path = os.path.join(_data_path, "JSBChorales.pickle")
            max_label = 96
        data = np.load(_path)
        raw = data[which_set]

        features = np.asarray(
                [np.asarray(
                    [self.list_to_nparray(time_step,
                    max_label) for time_step in np.asarray(raw[i][:-1])])
                    for i in xrange(len(raw))]
                    )
        targets = np.asarray(
            [np.asarray([self.list_to_nparray(time_step,
                max_label) for time_step in np.asarray(raw[i][1:])])
                for i in xrange(len(raw))]
            )
        return features, targets

    def list_to_nparray(self, x, dim):
        y = np.zeros((dim,), dtype=np.float32)
        for i in x:
            y[i - 1] = 1
        return y

    def get_data(self, state=None, request=None):
        #batch = next(state)
        batch = super(MidiSequence, self).get_data(state, request)
        #if state is not None:
        #batch = [b.transpose(1,0,2) for b in state[request]]
        return batch
