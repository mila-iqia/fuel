from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the binarized MNIST dataset files.

    The binarized MNIST dataset files
    (`binarized_mnist_{train,valid,test}.amat`) are downloaded from
    Hugo Larochelle's website [HUGO].

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `binarized_mnist` command.

    """
    sets = ['train', 'valid', 'test']
    urls = ['http://www.cs.toronto.edu/~larocheh/public/datasets/' +
            'binarized_mnist/binarized_mnist_{}.amat'.format(s) for s in sets]
    filenames = ['binarized_mnist_{}.amat'.format(s) for s in sets]
    subparser.set_defaults(urls=urls, filenames=filenames)
    return default_downloader
