from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging

from six import add_metaclass

from fuel.streams import AbstractDataStream
from ..exceptions import AxisLabelsMismatchError

log = logging.getLogger(__name__)


class ExpectsAxisLabels(object):
    """Mixin for transformers, used to verify axis labels.

    Notes
    -----
    Provides a method :meth:`verify_axis_labels` that should be called
    with the expected and actual values for an axis labels tuple. If
    `actual` is `None`, a warning is logged; if it is non-`None` and does
    not match `expected`, a :class:`AxisLabelsMismatchError` is raised.

    The check is only performed on the first call; if the call succeeds,
    an attribute is written to skip further checks, in the interest of
    speed.

    """
    def verify_axis_labels(self, expected, actual, source_name):
        """Verify that axis labels for a given source are as expected.

        Parameters
        ----------
        expected : tuple
            A tuple of strings representing the expected axis labels.
        actual : tuple or None
            A tuple of strings representing the actual axis labels, or
            `None` if they could not be determined.
        source_name : str
            The name of the source being checked. Used for caching the
            results of checks so that the check is only performed once.

        Notes
        -----
        Logs a warning in case of `actual=None`, raises an error on
        other mismatches.

        """
        if not getattr(self, '_checked_axis_labels', False):
            self._checked_axis_labels = defaultdict(bool)
        if not self._checked_axis_labels[source_name]:
            if actual is None:
                log.warning("%s instance could not verify (missing) axis "
                            "expected %s, got None",
                            self.__class__.__name__, expected)
            else:
                if expected != actual:
                    raise AxisLabelsMismatchError("{} expected axis labels "
                                                  "{}, got {} instead".format(
                        self.__class__.__name__,
                        expected, actual))
            self._checked_axis_labels[source_name] = True


@add_metaclass(ABCMeta)
class Transformer(AbstractDataStream):
    """A data stream that wraps another data stream.

    Subclasses must define a `transform_batch` method (to act on batches),
    a `transform_example` method (to act on individual examples), or
    both methods.

    Typically (using the interface mentioned above), the transformer
    is expected to have the same output type (example or batch) as its
    input type.  If the transformer subclass is going from batches to
    examples or vice versa, it should override `get_data` instead.
    Overriding `get_data` is also necessary when access to `request` is
    necessary (e.g. for the :class:`Cache` transformer).

    Attributes
    ----------
    child_epoch_iterator : iterator type
        When a new epoch iterator is requested, a new epoch creator is
        automatically requested from the wrapped data stream and stored in
        this attribute. Use it to access data from the wrapped data stream
        by calling ``next(self.child_epoch_iterator)``.
    produces_examples : bool
        Whether this transformer produces examples (as opposed to batches
        of examples).

    """
    def __init__(self, data_stream=None, produces_examples=None, **kwargs):
        self.kwargs = kwargs
        super(Transformer, self).__init__(**kwargs)
        if produces_examples is not None:
            self.produces_examples = produces_examples
        self.data_stream = data_stream

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.data_stream.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.data_stream.close()

    def reset(self):
        self.data_stream.reset()

    def next_epoch(self):
        self.data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the wrapped data set.

        Notes
        -----
        This default implementation assumes that the epochs of the wrapped
        data stream are less or equal in length to the original data
        stream. Implementations for which this is not true should request
        new epoch iterators from the child data set when necessary.

        """
        self.child_epoch_iterator = self.data_stream.get_epoch_iterator()
        return super(Transformer, self).get_epoch_iterator(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)

        if self.produces_examples != self.data_stream.produces_examples:
            types = {True: 'examples', False: 'batches'}
            raise NotImplementedError(
                "the wrapped data stream produces {} while the {} transformer "
                "produces {}, which it does not support.".format(
                    types[self.data_stream.produces_examples],
                    self.__class__.__name__,
                    types[self.produces_examples]))
        elif self.produces_examples:
            return self.transform_example(data)
        else:
            return self.transform_batch(data)

    def transform_example(self, example):
        """Transforms a single example."""
        raise NotImplementedError(
            "`{}` does not support examples as input, but the wrapped data "
            "stream produces examples.".format(self.__class__.__name__))

    def transform_batch(self, batch):
        """Transforms a batch of examples."""
        raise NotImplementedError(
            "`{}` does not support batches as input, but the wrapped data "
            "stream produces batches.".format(self.__class__.__name__))
