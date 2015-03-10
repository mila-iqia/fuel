Translation dataset
===================

For this example we will use the French-English parallel corpora from WMT15.
Start with downloading and unpacking them. Assuming you have ``FUEL_DATA_PATH``
set as an environment variable you can use the following set of commands:

.. code-block:: bash

   mkdir $FUEL_DATA_PATH/wmt15 && cd $_
   curl -O http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz \
        -O http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz \
        -O http://www.statmt.org/wmt13/training-parallel-un.tgz \
        -O http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz \
        -O http://www.statmt.org/wmt10/training-giga-fren.tar
   ls *.tgz | xargs -i tar xfvz {}
   tar xfv training-giga-fren.tar | xargs -i gunzip {}

Creating a dictionary
---------------------

Let's begin with creating a dictionary of English words and their frequency.

>>> files = ['commoncrawl.fr-en.en',
...          'news-commentary-v10.fr-en.en',
...          'giga-fren.release2.en',
...          'training/europarl-v7.fr-en.en',
...          'un/undoc.2000.fr-en.en']

We count the frequency of all words.

>>> from collections import Counter
>>> from itertools import chain
>>> counter = Counter()
>>> for filename in _files:
...     with open(filename) as f:
...         words = chain.from_iterable(line.split() for line in f)
...         counter.update(words)

Using the counter, we can easily create a dictionary. Reserve the first 3
indices for the unknown, beginning-of-sentence and end-of-sentence tokens.

.. tip::

   Since there are nearly 10 million tokens in the English WMT15 data, some of
   these commands might take a while. If you are using Python 2, it might be
   worthwhile using :class:`~itertools.izip` instead of :func:`zip` and
   :meth:`~collections.OrderedDict.iteritems` instead of
   :meth:`~collections.OrderedDict.items`. You should also consider using
   the ``protocol-pickle.HIGHEST_PROTOCOL`` flag for :meth:`pickle.dump`.

>>> from itertools import chain, count
>>> freq_words = list(zip(*counter.most_common()))[0]
>>> vocab = OrderedDict(zip(chain(['<UNK>', '<S>', '</S>'], freq_words), count()))

You might want to save the vocabulary so that we can easily re-use it later.

>>> with open('vocab.pkl', 'wb') as f:
...     pickle.dump(vocab, f)

The advantage of using a :class:`~collections.OrderedDict` is that we can
easily restrict the vocabulary to a given size. For example, to limit our
vocabulary to the 25,000 most frequent words (including the special ``UNK``,
``EOS`` and ``BOS`` tokens), we use:

>>> limited_vocab = OrderedDict(islice(vocab.items(), 25000))

Mege data streams
-----------------

Repeat the above process to create a dictionary of French words as well. Now
let's use the :class:`.TextFile` to create a dataset that will read the text
using the dictionary we just created.

>>> from fuel.datasets import TextFile
>>> dataset = TextFile(files, limited_vocab)
>>> stream = dataset.get_example_stream()
>>> next(stream.get_epoch_iterator())
([1, 1206, 34, 2399, 500, 19, 3157, 15, 4812, 48648, 2],)

We want to iterate over the two datasets simultaneously, so we merge them using
the :class:`.Merge` transformer.

>>> from fuel.transformers import Merge
>>> merged = Merge([en_stream, fr_stream], ('english', 'french'))

Batches of approximately uniform size
-------------------------------------

For efficiency reasons we want to train on minibatches of sentences that are
approximately equal in length. We accomplish this by reading a large number of
sentences into memory, sorting them by length, and then partioning this large
batch in smaller batches.

A stream of examples can be grouped into batches using the :class:`.Batch`
transformer.

>>> from fuel.transformers import Batch
>>> from fuel.schemes import ConstantScheme
>>> large_batches = Batch(merged, iteration_scheme=ConstantScheme(32 * 100))

We sort these batches using the :class:`.Mapping` operator in combination with
the :class:`.SortMapping`. Note that we can't pass a ``lambda`` function to the
:class:`.Mapping` transformer because of Python's serialization limitations.

>>> from fuel.transformers import Mapping, SortMapping
>>> def en_length(sentence_pair):
...     return len(sentence_pair[0])
>>> sorted_batches = Mapping(large_batches, SortMapping(en_length))

Splitting up the large batch into smaller batches can be done with the
:class:`.Cache` transformer.

>>> from fuel.transformers import Cache
>>> batches = Cache(sorted_batches, ConstantScheme(32))

For the final step we need to convert our sentences from ragged arrays to a
padded matrix and an accompanying mask.

>>> from fuel.transformers import Padding
>>> masked_batches = Padding(batches)

Reading in a separate process
-----------------------------

This entire pipeline which involves reading text from disk, sorting, padding,
etc. can be relatively slow. We can speed it up by doing all of this in a
separate process while our model is training. A simple way of doing this is
the :class:`.MultiProcessing` transformer.

>>> background_stream = MultiProcessing(masked_batches)

We can now use ``background_stream`` as any other stream, but in the background
it will already have 100 batches read, sorted and masked.
