from fuel.transformers import Transformer


class NGrams(Transformer):
    """Return n-grams from a stream.

    This data stream wrapper takes as an input a data stream outputting
    sentences. From these sentences n-grams of a fixed order (e.g. bigrams,
    trigrams, etc.) are extracted and returned. It also creates a
    ``targets`` data source. For each example, the target is the word
    immediately following that n-gram. It is normally used for language
    modeling, where we try to predict the next word from the previous *n*
    words.

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

    """
    def __init__(self, ngram_order, data_stream, target_source='targets',
                 **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        if len(data_stream.sources) > 1:
            raise ValueError
        super(NGrams, self).__init__(
            data_stream, produces_examples=True, **kwargs)
        self.sources = self.sources + (target_source,)
        self.ngram_order = ngram_order
        self.sentence = []
        self.index = 0

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        while not self.index < len(self.sentence) - self.ngram_order:
            self.sentence, = next(self.child_epoch_iterator)
            self.index = 0
        ngram = self.sentence[self.index:self.index + self.ngram_order]
        target = self.sentence[self.index + self.ngram_order]
        self.index += 1
        return (ngram, target)
