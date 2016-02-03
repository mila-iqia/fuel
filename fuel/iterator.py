import six


class DataIterator(six.Iterator):
    """An iterator over data, representing a single epoch.

    Parameters
    ----------
    data_stream : :class:`DataStream`
        The data stream over which to iterate.
    request_iterator : iterator
        An iterator which returns the request to pass to the data stream
        for each step.
    as_dict : bool, optional
        If `True`, return dictionaries mapping source names to data
        from each source. If `False` (default), return tuples in the
        same order as `data_stream.sources`.

    """
    def __init__(self, data_stream, request_iterator=None, as_dict=False):
        self.data_stream = data_stream
        self.request_iterator = request_iterator
        self.as_dict = as_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self.request_iterator is not None:
            data = self.data_stream.get_data(next(self.request_iterator))
        else:
            data = self.data_stream.get_data()
        if self.as_dict:
            return dict(zip(self.data_stream.sources, data))
        else:
            return data


class TransformerDataIterator(six.Iterator):
    """An iterator over transformed data, representing a single epoch.

    Parameters
    ----------
    transformer : :class:`Transformer`
        The transformer over which to iterate.
    child_iterator : :class:`DataIterator`
        The iterator of the transformer's data stream.
    request_iterator : iterator
        An iterator which returns the request to pass to the transformer
        for each step.
    as_dict : bool, optional
        If `True`, return dictionaries mapping source names to data
        from each source. If `False` (default), return tuples in the
        same order as `transformer.sources`.

    """
    def __init__(self, transformer, child_iterator, request_iterator=None,
                 as_dict=False):
        self.transformer = transformer
        self.child_iterator = child_iterator
        self.request_iterator = request_iterator
        self.as_dict = as_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self.request_iterator is not None:
            data = self.transformer.get_data(
                self.child_iterator, next(self.request_iterator))
        else:
            data = self.transformer.get_data(self.child_iterator)
        if self.as_dict:
            return dict(zip(self.transformer.sources, data))
        else:
            return data
