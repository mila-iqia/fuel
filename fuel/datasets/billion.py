import os

from fuel.datasets import TextFile
from fuel.utils import find_in_data_path


class OneBillionWord(TextFile):
    """Google's One Billion Word benchmark.

    This monolingual corpus contains 829,250,940 tokens (including sentence
    boundary markers). The data is split into 100 partitions, one of which
    is the held-out set. This held-out set is further divided into 50
    partitions. More information about the dataset can be found in
    [CMSG14].

    .. [CSMG14] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, and
       Thorsten Brants, *One Billion Word Benchmark for Measuring Progress
       in Statistical Language Modeling*, `arXiv:1312.3005 [cs.CL]
       <http://arxiv.org/abs/1312.3005>`.

    Parameters
    ----------
    which_set : 'training' or 'heldout'
        Which dataset to load.
    which_partitions : list of ints
        For the training set, valid values must lie in [1, 99]. For the
        heldout set they must be in [0, 49].
    vocabulary : dict
        A dictionary mapping tokens to integers. This dictionary is
        expected to contain the tokens ``<S>``, ``</S>`` and ``<UNK>``,
        representing "start of sentence", "end of sentence", and
        "out-of-vocabulary" (OoV). The latter will be used whenever a token
        cannot be found in the vocabulary.
    preprocess : function, optional
        A function that takes a string (a sentence including new line) as
        an input and returns a modified string. A useful function to pass
        could be ``str.lower``.

    See :class:`TextFile` for remaining keyword arguments.

    """
    def __init__(self, which_set, which_partitions, dictionary, **kwargs):
        if which_set not in ('training', 'heldout'):
            raise ValueError
        if which_set == 'training':
            if not all(partition in range(1, 100)
                       for partition in which_partitions):
                raise ValueError
            files = [find_in_data_path(os.path.join(
                '1-billion-word', 'training-monolingual.tokenized.shuffled',
                'news.en-{:05d}-of-00100'.format(partition)))
                for partition in which_partitions]
        else:
            if not all(partition in range(50)
                       for partition in which_partitions):
                raise ValueError
            files = [find_in_data_path(os.path.join(
                '1-billion-word', 'heldout-monolingual.tokenized.shuffled',
                'news.en.heldout-{:05d}-of-00050'.format(partition)))
                for partition in which_partitions]
        super(OneBillionWord, self).__init__(files, dictionary, **kwargs)
