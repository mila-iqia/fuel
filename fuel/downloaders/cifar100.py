from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the CIFAR-100 dataset file.

    The CIFAR-100 dataset file is downloaded from Alex Krizhevsky's
    website [ALEX].

    .. [ALEX] http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar100` command.

    """
    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    subparser.set_defaults(urls=[url], filenames=[filename])
    return default_downloader
