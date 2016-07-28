from abc import ABCMeta
from six import add_metaclass

from .base import Transformer


@add_metaclass(ABCMeta)
class Pipe(Transformer):
    def __or__(self, other):
        print("call or")
        return CompositePipe(self, other)

    def __lt__(self, other):
        print("call lt")
        self.data_stream = other
        return self


class CompositePipe(Pipe):
    def __init__(self, first, second, **kwargs):
        self.first = first
        self.second = second
        super(CompositePipe, self).__init__(None, **kwargs)

    @property
    def produces_examples(self):
        return self.second.produces_examples

    @property
    def data_stream(self):
        return self._data_stream

    @data_stream.setter
    def data_stream(self, stream):
        self._data_stream = stream
        self.first.data_stream = stream
        self.second.data_stream = self.first

    def transform_example(self, example):
        """Transforms a single example."""
        return self.second.transorm_example(
            self.first.transform_example(example))

    def transform_batch(self, batch):
        """Transforms a batch of examples."""
        return self.second.transform_batch(self.first.transform_batch(batch))
