import numpy as np

from fuel.datasets.midi import MidiSequence
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Padding, Mapping

def test_jsb():
    jsb = MidiSequence('jsb')
    print jsb.num_examples
    dataset = DataStream(
                jsb,
                iteration_scheme=SequentialScheme(
                               jsb.num_examples, 10
                )
              )
    for b in dataset.get_epoch_iterator():
        print b[0].shape

    # This is how to prepare this dataset to be used in Blocks
    dataset = Padding(dataset)
    def _transpose(data):
        return tuple(np.rollaxis(array,1,0) for array in data)
    dataset = Mapping(dataset, _transpose)

    for b in dataset.get_epoch_iterator():
        print len(b)
        print b[1].shape
        print b[0].shape

    assert b[0].shape == (108,9,96)
if __name__ == '__main__':
    test_jsb()
