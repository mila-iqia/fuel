from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the MNIST dataset files.

    The following MNIST dataset files are downoladed from Yann LeCun's
    website [LECUN]:
    `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
    `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.

    .. [LECUN] http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    urls = ['http://yann.lecun.com/exdb/mnist/' + f for f in filenames]
    subparser.set_defaults(
        func=default_downloader, urls=urls, filenames=filenames)
