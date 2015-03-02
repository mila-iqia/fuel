from fuel.datasets import IterableDataset
from fuel.transformers import Mapping, MultiProcessing


def test_multiprocessing():
    stream = IterableDataset(range(100)).get_example_stream()
    plus_one = Mapping(stream, lambda x: (x[0] + 1,))
    background = MultiProcessing(plus_one)
    for a, b in zip(background.get_epoch_iterator(), range(1, 101)):
        assert a == (b,)
