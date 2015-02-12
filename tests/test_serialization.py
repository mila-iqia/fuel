import os
import tempfile

from six.moves import cPickle
import numpy

from fuel.streams import DataStream
from fuel.mnist import MNIST
from fuel.schemes import SequentialScheme


def test_in_memory():
    # Load MNIST and get two batches
    mnist = MNIST('train')
    data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
        num_examples=mnist.num_examples, batch_size=256))
    epoch = data_stream.get_epoch_iterator()
    for i, (features, targets) in enumerate(epoch):
        if i == 1:
            break
    assert numpy.all(features == mnist.features[256:512])

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
    assert numpy.all(features == mnist.features[512:768])
