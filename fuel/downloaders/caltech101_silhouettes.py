from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the CalTech101 Silhouettes dataset files.

    The following MNIST dataset files are downoladed from Benjamin M. Marlin's
    website [MARLIN]:
        `caltech101_silhouettes_16_split1.mat' and `caltech101_silhouettes_28_split1.mat`.

    .. [MARLIN] https://people.cs.umass.edu/~marlin/data.shtml

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    filenames = ['caltech101_silhouettes_16_split1.mat', 
                 'caltech101_silhouettes_28_split1.mat']
    urls = ['https://people.cs.umass.edu/~marlin/data/' + f for f in filenames]
    subparser.set_defaults(
        func=default_downloader, urls=urls, filenames=filenames)
