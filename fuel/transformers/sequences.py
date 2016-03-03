from fuel.transformers import Transformer


class Window(Transformer):
    """Return pairs of source and target windows from a stream.

    This data stream wrapper takes as an input a data stream outputting
    sequences of potentially varying lengths (e.g. sentences, audio tracks,
    etc.). It then returns two sliding windows (source and target) over
    these sequences.

    For example, to train an n-gram model set `source_window` to n,
    `target_window` to 1, no offset, and `overlapping` to false. This will
    give chunks [1, N] and [N + 1]. To train an RNN you often want to set
    the source and target window to the same size and use an offset of 1
    with overlap, this would give you chunks [1, N] and [2, N + 1].

    Parameters
    ----------
    offset : int
        The offset from the source window where the target window starts.
    source_window : int
        The size of the source window.
    target_window : int
        The size of the target window.
    overlapping : bool
        If true, the source and target windows overlap i.e. the offset of
        the target window is taken to be from the beginning of the source
        window. If false, the target window offset is taken to be from the
        end of the source window.
    data_stream : :class:`.DataStream` instance
        The data stream providing sequences. Each example is assumed to be
        an object that supports slicing.
    target_source : str, optional
        This data stream adds a new source for the target words. By default
        this source is 'targets'.

    """
    def __init__(self, offset, source_window, target_window,
                 overlapping, data_stream, target_source='targets', **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        if len(data_stream.sources) > 1:
            raise ValueError('{} expects only one source'
                             .format(self.__class__.__name__))

        super(Window, self).__init__(data_stream, produces_examples=True,
                                     **kwargs)
        self.sources = self.sources + (target_source,)

        self.offset = offset
        self.source_window = source_window
        self.target_window = target_window
        self.overlapping = overlapping

        self.sentence = []
        self._set_index()

    def _set_index(self):
        """Set the starting index of the source window."""
        self.index = 0
        # If offset is negative, target window might start before 0
        self.index = -min(0, self._get_target_index())

    def _get_target_index(self):
        """Return the index where the target window starts."""
        return (self.index + self.source_window * (not self.overlapping) +
                self.offset)

    def _get_end_index(self):
        """Return the end of both windows."""
        return max(self.index + self.source_window,
                   self._get_target_index() + self.target_window)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        while not self._get_end_index() <= len(self.sentence):
            self.sentence, = next(self.child_epoch_iterator)
            self._set_index()
        source = self.sentence[self.index:self.index + self.source_window]
        target = self.sentence[self._get_target_index():
                               self._get_target_index() + self.target_window]
        self.index += 1
        return (source, target)


class NGrams(Window):
    """Return n-grams from a stream.

    This data stream wrapper takes as an input a data stream outputting
    sentences. From these sentences n-grams of a fixed order (e.g. bigrams,
    trigrams, etc.) are extracted and returned. It also creates a
    ``targets`` data source. For each example, the target is the word
    immediately following that n-gram. It is normally used for language
    modeling, where we try to predict the next word from the previous *n*
    words.

    .. note::

       Unlike the :class:`Window` stream, the target returned by
       :class:`NGrams` is a single element instead of a window.

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
    def __init__(self, ngram_order, *args, **kwargs):
        super(NGrams, self).__init__(
            0, ngram_order, 1, False, *args, **kwargs)

    def get_data(self, *args, **kwargs):
        source, target = super(NGrams, self).get_data(*args, **kwargs)
        return (source, target[0])
