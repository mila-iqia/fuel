from fuel.downloaders.base import default_downloader


def svhn_downloader(which_format, directory, clear=False):
    suffix = {1: '.tar.gz', 2: '_32x32.mat'}[which_format]
    sets = ['train', 'test', 'extra']
    default_downloader(
        directory=directory,
        urls=[None for f in sets],
        filenames=['{}{}'.format(s, suffix) for s in sets],
        url_prefix='http://ufldl.stanford.edu/housenumbers/',
        clear=clear)


def fill_subparser(subparser):
    """Sets up a subparser to download the SVHN dataset files.

    The SVHN dataset files (`{train,test,extra}{.tar.gz,_32x32.mat}`)
    are downloaded from the official website [SVHNSITE].

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `svhn` command.

    """
    subparser.add_argument(
        "which_format", help="which dataset format", type=int, choices=(1, 2))
    return svhn_downloader
