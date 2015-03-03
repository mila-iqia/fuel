import os
from collections import OrderedDict

import numpy
import scipy
from scipy.io import matlab
import six

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes

list_sets = "SVHN only has a train, extra, full_train and test set."


@do_not_pickle_attributes('indexables')
class SVHN(IndexableDataset):
    """The Street View House Numbers (SVHN) Dataset. Format 2: Cropped Digits.

    SVHN is obtained from house numbers in Google Street View images in
    format 2. This format consists of 32 x 32 colour images in 10 classes,
    1 for each digit: 73,257 digits for training, 26,032 digits for testing,
    and 531,131 additional, somewhat less difficult samples, to use as extra
    training data [SVHN]. The extra training data and the regular training data
    can be merged as `full_train`.

    .. [SVHN] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu
        and Andrew Y. Ng, *Reading Digits in Natural Images with Unsupervised
        Feature Learning*, NIPS Workshop on Deep Learning and Unsupervised
        Feature Learning, 2011

    Parameters
    ----------
    which_set : 'train', 'extra', 'full_train' or 'test'
        Whether to load the training set (73,257 samples), the extra training
        set (531,131 samples), a union of the two latter (604,388 samples) or
        the test set (26,032 samples). Note that SVHN does not have a
        validation set; usually you will create your own training/validation
        split using the start and stop arguments.
    flatten : bool
        Whether to flatten the images. If ``False``, returns images of the
        format (3, 32, 32). Is ``True`` by default.

    """
    provides_sources = ('features', 'targets')
    folder = os.path.join('SVHN', 'format2')
    files = {
        'train': ['train_32x32.mat'],
        'extra': ['extra_32x32.mat'],
        'test': ['test_32x32.mat'],
        'full_train': ['train_32x32.mat', 'extra_32x32.mat']
    }

    def __init__(self, which_set, flatten=True, **kwargs):
        if which_set not in self.files.keys():
            raise ValueError("Wrong set. " + list_sets)

        self.which_set = which_set
        self.flatten = flatten

        super(SVHN, self).__init__(OrderedDict(zip(self.provides_sources,
                                                   self._load_svhn())),
                                   **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_svhn())
                           if source in self.sources]

    def _load_svhn(self):
        base_path = os.path.join(config.data_path, self.folder)
        num_examples = {
            'train': 73257, 'extra': 531131,
            'full_train': 604388, 'test': 26032
        }[self.which_set]
        image_shape = (3072,) if self.flatten else (3, 32, 32)
        images = numpy.zeros((num_examples,) + image_shape,
                             dtype=config.floatX)
        labels = numpy.zeros((num_examples, 1), dtype='uint8')

        current = 0
        for fname in self.files[self.which_set]:
            with open(os.path.join(base_path, fname), 'rb') as f:
                batch = matlab.loadmat(os.path.join(base_path, fname))
                batch['X'] = batch['X'].transpose((3, 2, 0, 1))
                next_current = current + batch['X'].shape[0]
                if self.flatten:
                    batch['X'] = batch['X'].reshape(
                        (batch['X'].shape[0], 3072)
                    )
                images[current:next_current] = batch['X']
                labels[current:next_current] = numpy.asarray(
                    batch['y'] - 1, dtype='uint8')
                current = next_current
        return images, labels
