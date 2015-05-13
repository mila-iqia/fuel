from numpy.testing import assert_raises

from fuel.datasets.billion import OneBillionWord


class TestOneBillionWord(object):
    def setUp(self):
        all_chars = ([chr(ord('a') + i) for i in range(26)] +
                     [chr(ord('0') + i) for i in range(10)] +
                     [',', '.', '!', '?', '<UNK>'] +
                     [' ', '<S>', '</S>'])
        code2char = dict(enumerate(all_chars))
        self.char2code = {v: k for k, v in code2char.items()}

    def test_value_error_wrong_set(self):
        assert_raises(
            ValueError, OneBillionWord, 'dummy',  [0, 1], self.char2code)

    def test_value_error_training_partition(self):
        assert_raises(
            ValueError, OneBillionWord, 'training',  [101], self.char2code)

    def test_value_error_heldout_partition(self):
        assert_raises(
            ValueError, OneBillionWord, 'heldout',  [101], self.char2code)
