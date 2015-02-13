import numpy
from toolz import sliding_window

from fuel.streams import CachedDataStream


class NGramStream(CachedDataStream):
    """Return n-grams from a stream.

    This data stream wrapper takes as an input a data stream outputting
    batches of sentences. From these sentences n-grams of a fixed order
    (e.g. bigrams, trigrams, etc.) are extracted and returned. It also
    creates a ``targets`` data source. For each example, the target is the
    word immediately following that n-gram. It is normally used for
    language modeling, where we try to predict the next word from the
    previous n words.

    Parameters
    ----------
    ngram_order : int
        The order of the n-grams to output e.g. 3 for trigrams.
    data_stream : :class:`.DataStream` instance
        The data stream providing sentences. Each example is assumed to be
        a list of integers.
    target_source : str, optional
        This data stream adds a new source for the target words. By default
        this source is 'targets'.

    Notes
    -----
    This class inherits from :class:`.CachedDataStream` because it makes
    use of a cache to store the sentences from the wrapped data stream in.

    """
    def __init__(self, ngram_order, data_stream, target_source='targets',
                 iteration_scheme=None):
        if len(data_stream.sources) > 1:
            raise ValueError
        super(NGramStream, self).__init__(data_stream, iteration_scheme)
        self.sources = self.sources + (target_source,)
        self.ngram_order = ngram_order

    def get_data(self, request=None):
        features, targets = [], []
        for _, sentence in enumerate(self.cache[0]):
            features.append(list(
                sliding_window(self.ngram_order,
                               sentence[:-1]))[:request - len(features)])
            targets.append(
                sentence[self.ngram_order:][:request - len(targets)])
            self.cache[0][0] = self.cache[0][0][request:]
            if not self.cache[0][0]:
                self.cache[0].pop(0)
                if not self.cache[0]:
                    self._cache()
            if len(features) == request:
                break
        return tuple(numpy.asarray(data) for data in (features, targets))
