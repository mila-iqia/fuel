import os
from subprocess import call

import fuel

default_save_path = os.path.join(fuel.config.data_path, 'binarized_mnist')


def binarized_mnist(save_directory):
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

    """
    base_url = ('http://www.cs.toronto.edu/~larocheh/public/datasets/' +
                'binarized_mnist/binarized_mnist_')
    call(['curl', base_url + 'train.amat', '-o',
          os.path.join(save_directory, 'binarized_mnist_train.amat')])
    call(['curl', base_url + 'valid.amat', '-o',
          os.path.join(save_directory, 'binarized_mnist_valid.amat')])
    call(['curl', base_url + 'test.amat', '-o',
          os.path.join(save_directory, 'binarized_mnist_test.amat')])
