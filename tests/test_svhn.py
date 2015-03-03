import numpy
from numpy.testing import assert_raises
from six.moves import cPickle

from fuel.datasets import SVHN


def test_svhn():
    svhn_train = SVHN('train', start=20000)
    assert len(svhn_train.features) == 53257
    assert len(svhn_train.targets) == 53257
    assert svhn_train.num_examples == 53257
    svhn_extra = SVHN('extra', start=10000)
    assert len(svhn_extra.features) == 521131
    assert len(svhn_extra.targets) == 521131
    assert svhn_extra.num_examples == 521131
    svhn_full_train = SVHN('full_train', start=1000)
    assert len(svhn_full_train.features) == 603388
    assert len(svhn_full_train.targets) == 603388
    assert svhn_full_train.num_examples == 603388
    svhn_test = SVHN('test', sources=('targets',))
    assert len(svhn_test.targets) == 26032
    assert svhn_test.num_examples == 26032

    first_feature, first_target = svhn_train.get_data(request=[0])
    assert first_feature.shape == (1, 3072)
    assert first_feature.dtype.kind == 'f'
    assert first_target.shape == (1, 1)
    assert first_target.dtype is numpy.dtype('uint8')

    first_target, = svhn_test.get_data(request=[0, 1])
    assert first_target.shape == (2, 1)

    assert_raises(ValueError, SVHN, 'valid')

    svhn_test = cPickle.loads(cPickle.dumps(svhn_test))
    assert len(svhn_test.targets) == 26032

    cifar_10_test_unflattened = SVHN('test', flatten=False)
    cifar_10_test_unflattened.features.shape == (26032, 3, 32, 32)
