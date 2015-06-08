from fuel.downloaders.base import default_downloader


base_url = 'https://people.cs.umass.edu/~marlin/data/'
filename = 'caltech101_silhouettes_{}_split1.mat'


def caltech101_silhouettes_downloader(size, **kwargs):
    if size not in [16, 28]:
        raise ValueError("size must be 16 or 28")

    actual_filename = filename.format(size)
    actual_url = base_url + actual_filename
    return default_downloader(urls=[actual_url],
                              filenames=[actual_filename], **kwargs)


def fill_subparser(subparser):
    """Sets up a subparser to download the CalTech101 Silhouettes dataset files.

    The following MNIST dataset files can be downloaded from Benjamin
    M. Marlin's website [MARLIN]:
        `caltech101_silhouettes_16_split1.mat` and
        `caltech101_silhouettes_28_split1.mat`.

    .. [MARLIN] https://people.cs.umass.edu/~marlin/data.shtml

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    subparser.add_argument(
        "size", type=int, choices=(16, 28),
        help="height/width of the datapoints")
    subparser.set_defaults(
        func=caltech101_silhouettes_downloader)
