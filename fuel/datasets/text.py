from picklable_itertools import iter_, chain

from fuel.datasets import Dataset
from fuel.utils.formats import open_


class TextFile(Dataset):
    r"""Reads text files and numberizes them given a dictionary.

    Parameters
    ----------
    files : list of str
        The names of the files in order which they should be read. Each
        file is expected to have a sentence per line. If the filename ends
        with `.gz` it will be opened using `gzip`. Note however that `gzip`
        file handles aren't picklable on legacy Python.
    dictionary : str or dict
        Either the path to a Pickled dictionary mapping tokens to integers,
        or the dictionary itself. At the very least this dictionary must
        map the unknown word-token to an integer.
    bos_token : str or None, optional
        The beginning-of-sentence (BOS) token in the dictionary that
        denotes the beginning of a sentence. Is ``<S>`` by default. If
        passed ``None`` no beginning of sentence markers will be added.
    eos_token : str or None, optional
        The end-of-sentence (EOS) token is ``</S>`` by default, see
        ``bos_taken``.
    unk_token : str, optional
        The token in the dictionary to fall back on when a token could not
        be found in the dictionary. ``<UNK>`` by default. Pass ``None`` if
        the dataset doesn't contain any out-of-vocabulary words/characters
        (the data request is going to crash if meets an unknown symbol).

    level : 'word' or 'character', optional
        If 'word' the dictionary is expected to contain full words. The
        sentences in the text file will be split at the spaces, and each
        word replaced with its number as given by the dictionary, resulting
        in each example being a single list of numbers. If 'character' the
        dictionary is expected to contain single letters as keys. A single
        example will be a list of character numbers, starting with the
        first non-whitespace character and finishing with the last one. The
        default is 'word'.
    preprocess : function, optional
        A function which takes a sentence (string) as an input and returns
        a modified string. For example ``str.lower`` in order to lowercase
        the sentence before numberizing.
    encoding : str, optional
        The encoding to use to read the file. Defaults to ``None``. Use
        UTF-8 if the dictionary you pass contains UTF-8 characters, but
        note that this makes the dataset unpicklable on legacy Python.

    Examples
    --------
    >>> with open('sentences.txt', 'w') as f:
    ...     _ = f.write("This is a sentence\n")
    ...     _ = f.write("This another one")
    >>> dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    >>> def lower(s):
    ...     return s.lower()
    >>> text_data = TextFile(files=['sentences.txt'],
    ...                      dictionary=dictionary, bos_token=None,
    ...                      preprocess=lower)
    >>> from fuel.streams import DataStream
    >>> for data in DataStream(text_data).get_epoch_iterator():
    ...     print(data)
    ([2, 0, 3, 0, 1],)
    ([2, 0, 4, 1],)
    >>> full_dictionary = {'this': 0, 'a': 3, 'is': 4, 'sentence': 5,
    ...                    'another': 6, 'one': 7}
    >>> text_data = TextFile(files=['sentences.txt'],
    ...                      dictionary=full_dictionary, bos_token=None,
    ...                      eos_token=None, unk_token=None,
    ...                      preprocess=lower)
    >>> for data in DataStream(text_data).get_epoch_iterator():
    ...     print(data)
    ([0, 4, 3, 5],)
    ([0, 6, 7],)

    .. doctest::
       :hide:

       >>> import os
       >>> os.remove('sentences.txt')

    """
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None,
                 encoding=None):
        self.files = files
        self.dictionary = dictionary
        if bos_token is not None and bos_token not in dictionary:
            raise ValueError(
                "BOS token '{}' is not in the dictionary".format(bos_token))
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in dictionary:
            raise ValueError(
                "EOS token '{}' is not in the dictionary".format(eos_token))
        self.eos_token = eos_token
        if unk_token is not None and unk_token not in dictionary:
            raise ValueError(
                "UNK token '{}' is not in the dictionary".format(unk_token))
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError(
                "level should be 'word' or 'character', not '{}'"
                .format(level))
        self.level = level
        self.preprocess = preprocess
        self.encoding = encoding
        super(TextFile, self).__init__()

    def open(self):
        return chain(*[iter_(open_(f, encoding=self.encoding))
                       for f in self.files])

    def _get_from_dictionary(self, symbol):
        value = self.dictionary.get(symbol)
        if value is not None:
            return value
        else:
            if self.unk_token is None:
                raise KeyError("token '{}' not found in dictionary and no "
                               "`unk_token` given".format(symbol))
            return self.dictionary[self.unk_token]

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        sentence = next(state)
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(self._get_from_dictionary(word)
                        for word in sentence.split())
        else:
            data.extend(self._get_from_dictionary(char)
                        for char in sentence.strip())
        if self.eos_token:
            data.append(self.dictionary[self.eos_token])
        return (data,)
