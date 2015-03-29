import os
import shutil
from subprocess import call

import fuel
from six.moves.urllib.request import urlopen

default_save_path = os.path.join(fuel.config.data_path, 'binarized_mnist')


def binarized_mnist(save_directory, clear=False):
    """Downloads the binarized MNIST dataset files.

    The binarized MNIST dataset files
    (`binarized_mnist_{train,valid,test}.amat`) are downloaded from
    Hugo Larochelle's website [HUGO].

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    save_directory : str
        Where to save the downloaded files.
    clear : bool, optional
        If `True`, clear the downloaded files. Defaults to `False`.

    """
    train_file = os.path.join(save_directory, 'binarized_mnist_train.amat')
    valid_file = os.path.join(save_directory, 'binarized_mnist_valid.amat')
    test_file = os.path.join(save_directory, 'binarized_mnist_test.amat')
    if clear:
        if os.path.isfile(train_file):
            os.remove(train_file)
        if os.path.isfile(valid_file):
            os.remove(valid_file)
        if os.path.isfile(test_file):
            os.remove(test_file)
    else:
        base_url = ('http://www.cs.toronto.edu/~larocheh/public/datasets/' +
                    'binarized_mnist/binarized_mnist_')
        response = urlopen(base_url + 'train.amat')
        with open(train_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        response.close()
        response = urlopen(base_url + 'valid.amat')
        with open(valid_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        response.close()
        response = urlopen(base_url + 'test.amat')
        with open(test_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        response.close()
