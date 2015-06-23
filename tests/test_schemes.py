import numpy
from numpy.testing import assert_raises

from fuel.schemes import (ConstantScheme, SequentialExampleScheme,
                          SequentialScheme, ShuffledExampleScheme,
                          ShuffledScheme, ConcatenatedScheme,
                          cross_validation)


def iterator_requester(scheme):
    def get_request_iterator(*args, **kwargs):
        scheme_obj = scheme(*args, **kwargs)
        return scheme_obj.get_request_iterator()
    return get_request_iterator


def test_constant_scheme():
    get_request_iterator = iterator_requester(ConstantScheme)
    assert list(get_request_iterator(3, num_examples=7)) == [3, 3, 1]
    assert list(get_request_iterator(3, num_examples=9)) == [3, 3, 3]
    assert list(get_request_iterator(3, num_examples=2)) == [2]
    assert list(get_request_iterator(2, times=3)) == [2, 2, 2]
    assert list(get_request_iterator(3, times=1)) == [3]
    it = get_request_iterator(3)
    assert [next(it) == 3 for _ in range(10)]
    assert_raises(ValueError, get_request_iterator, 10, 2, 2)


def test_sequential_scheme():
    get_request_iterator = iterator_requester(SequentialScheme)
    assert list(get_request_iterator(5, 3)) == [[0, 1, 2], [3, 4]]
    assert list(get_request_iterator(4, 2)) == [[0, 1], [2, 3]]
    assert list(get_request_iterator(
        [4, 3, 2, 1, 0], 3)) == [[4, 3, 2], [1, 0]]
    assert list(get_request_iterator(
        [3, 2, 1, 0], 2)) == [[3, 2], [1, 0]]


def test_shuffled_scheme_sorted_indices():
    get_request_iterator = iterator_requester(ShuffledScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=True)) == \
        [sorted(indices[:3]), sorted(indices[3:6]), sorted(indices[6:])]
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=True)) != \
        [sorted(indices[:3]), sorted(indices[3:6]), sorted(indices[6:])]

    indices = list(range(6))[::-1]
    expected = indices[:]
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(expected)
    assert (list(get_request_iterator(indices, 3, rng=rng,
                                      sorted_indices=True)) ==
            [sorted(expected[:3]), sorted(expected[3:6])])


def test_shuffled_scheme_unsorted_indices():
    get_request_iterator = iterator_requester(ShuffledScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=False)) == \
        [indices[:3], indices[3:6], indices[6:]]
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=False)) != \
        [indices[:3], indices[3:6], indices[6:]]

    indices = list(range(6))[::-1]
    expected = indices[:]
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(expected)
    assert (list(get_request_iterator(indices, 3, rng=rng,
                                      sorted_indices=False)) ==
            [expected[:3], expected[3:6]])


def test_shuffled_example_scheme():
    get_request_iterator = iterator_requester(ShuffledExampleScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, rng=rng)) == indices


def test_shuffled_example_scheme_no_rng():
    scheme = ShuffledExampleScheme(7)
    assert scheme.rng is not None


def test_sequential_example_scheme():
    get_request_iterator = iterator_requester(SequentialExampleScheme)
    assert list(get_request_iterator(7)) == list(range(7))
    assert list(get_request_iterator(range(7)[::-1])) == list(range(7)[::-1])


def test_concatenated_scheme():
    sch = ConcatenatedScheme(schemes=[ConstantScheme(batch_size=10, times=5),
                                      ConstantScheme(batch_size=20, times=3),
                                      ConstantScheme(batch_size=30, times=1)])
    assert (list(sch.get_request_iterator()) ==
            ([10] * 5) + ([20] * 3) + [30])


def test_cross_validation():
    # test raise when strict=True
    cross = cross_validation(SequentialExampleScheme, 10, 3)
    assert_raises(ValueError, next, cross)

    # test IndexScheme when strict=False
    cross = cross_validation(SequentialExampleScheme, 10, 3, False)

    (train, valid, valid_size) = next(cross)
    assert list(train.get_request_iterator()) == list(range(3, 10))
    assert list(valid.get_request_iterator()) == list(range(0, 3))
    assert valid_size == 3

    (train, valid, valid_size) = next(cross)
    assert (list(train.get_request_iterator()) ==
            list(range(0, 3)) + list(range(6, 10)))
    assert list(valid.get_request_iterator()) == list(range(3, 6))
    assert valid_size == 3

    (train, valid, valid_size) = next(cross)
    assert list(train.get_request_iterator()) == list(range(0, 6))
    assert list(valid.get_request_iterator()) == list(range(6, 10))
    assert valid_size == 4

    assert_raises(StopIteration, next, cross)

    # test BatchScheme
    cross = cross_validation(SequentialScheme, 8, 2, batch_size=2)

    (train, valid) = next(cross)
    assert list(train.get_request_iterator()) == [[4, 5], [6, 7]]
    assert list(valid.get_request_iterator()) == [[0, 1], [2, 3]]

    (train, valid) = next(cross)
    assert list(train.get_request_iterator()) == [[0, 1], [2, 3]]
    assert list(valid.get_request_iterator()) == [[4, 5], [6, 7]]

    assert_raises(StopIteration, next, cross)
