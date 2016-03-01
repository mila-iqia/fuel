import tempfile

import numpy
from numpy.testing import assert_raises
from six import BytesIO
from six.moves import cPickle

from fuel.datasets import TextFile, IterableDataset, IndexableDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers.sequences import Window, NGrams


def lower(s):
    return s.lower()


def test_text():
    # Test word level and epochs.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        sentences1 = f.name
        f.write("This is a sentence\n")
        f.write("This another one")
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        sentences2 = f.name
        f.write("More sentences\n")
        f.write("The last one")
    dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    text_data = TextFile(files=[sentences1, sentences2],
                         dictionary=dictionary, bos_token=None,
                         preprocess=lower)
    stream = DataStream(text_data)
    epoch = stream.get_epoch_iterator()
    assert len(list(epoch)) == 4
    epoch = stream.get_epoch_iterator()
    for sentence in zip(range(3), epoch):
        pass
    f = BytesIO()
    cPickle.dump(epoch, f)
    sentence = next(epoch)
    f.seek(0)
    epoch = cPickle.load(f)
    assert next(epoch) == sentence
    assert_raises(StopIteration, next, epoch)

    # Test character level.
    dictionary = dict([(chr(ord('a') + i), i) for i in range(26)] +
                      [(' ', 26)] + [('<S>', 27)] +
                      [('</S>', 28)] + [('<UNK>', 29)])
    text_data = TextFile(files=[sentences1, sentences2],
                         dictionary=dictionary, preprocess=lower,
                         level="character")
    sentence = next(DataStream(text_data).get_epoch_iterator())[0]
    assert sentence[:3] == [27, 19, 7]
    assert sentence[-3:] == [2, 4, 28]


def test_ngram_stream():
    sentences = [list(numpy.random.randint(10, size=sentence_length))
                 for sentence_length in [3, 5, 7]]
    stream = DataStream(IterableDataset(sentences))
    ngrams = NGrams(4, stream)
    assert len(list(ngrams.get_epoch_iterator())) == 4


def test_window_stream():
    sentences = [list(numpy.random.randint(10, size=sentence_length))
                 for sentence_length in [3, 5, 7]]
    stream = DataStream(IterableDataset(sentences))
    windows = Window(0, 4, 4, True, stream)
    for i, (source, target) in enumerate(windows.get_epoch_iterator()):
        assert source == target
    assert i == 5  # Total of 6 windows

    # Make sure that negative indices work
    windows = Window(-2, 4, 4, False, stream)
    for i, (source, target) in enumerate(windows.get_epoch_iterator()):
        assert source[-2:] == target[:2]
    assert i == 1  # Should get 2 examples

    # Even for overlapping negative indices should work
    windows = Window(-2, 4, 4, True, stream)
    for i, (source, target) in enumerate(windows.get_epoch_iterator()):
        assert source[:2] == target[-2:]
    assert i == 1  # Should get 2 examples


def test_ngram_stream_error_on_multiple_sources():
    # Check that NGram accepts only data streams with one source
    sentences = [list(numpy.random.randint(10, size=sentence_length))
                 for sentence_length in [3, 5, 7]]
    stream = DataStream(IterableDataset(sentences))
    stream.sources = ('1', '2')
    assert_raises(ValueError, NGrams, 4, stream)


def test_ngram_stream_raises_error_on_batch_stream():
    sentences = [list(numpy.random.randint(10, size=sentence_length))
                 for sentence_length in [3, 5, 7]]
    stream = DataStream(
        IndexableDataset(sentences), iteration_scheme=SequentialScheme(3, 1))
    assert_raises(ValueError, NGrams, 4, stream)


def test_ngram_stream_raises_error_on_request():
    sentences = [list(numpy.random.randint(10, size=sentence_length))
                 for sentence_length in [3, 5, 7]]
    stream = DataStream(IterableDataset(sentences))
    ngrams = NGrams(4, stream)
    assert_raises(ValueError, ngrams.get_data, [0, 1])
