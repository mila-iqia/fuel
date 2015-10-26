import os
import tempfile

import numpy
from six.moves import cPickle

from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from tests import skip_if_not_available


def test_in_memory():
    skip_if_not_available(datasets=['mnist.hdf5'])
    # Load MNIST and get two batches
    mnist = MNIST(('train',), load_in_memory=True)
    data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
        examples=mnist.num_examples, batch_size=256))
    epoch = data_stream.get_epoch_iterator()
    for i, (features, targets) in enumerate(epoch):
        if i == 1:
            break
    handle = mnist.open()
    known_features, _ = mnist.get_data(handle, slice(256, 512))
    mnist.close(handle)
    assert numpy.all(features == known_features)

    # Pickle the epoch and make sure that the data wasn't dumped
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        cPickle.dump(epoch, f)
    assert os.path.getsize(filename) < 1024 * 1024  # Less than 1MB

    # Reload the epoch and make sure that the state was maintained
    del epoch
    with open(filename, 'rb') as f:
        epoch = cPickle.load(f)
    features, targets = next(epoch)
    handle = mnist.open()
    known_features, _ = mnist.get_data(handle, slice(512, 768))
    mnist.close(handle)
    assert numpy.all(features == known_features)
