from fuel.downloaders.base import default_downloader


BASE_URL = 'https://people.cs.umass.edu/~marlin/data/'
FILENAME = 'caltech101_silhouettes_{}_split1.mat'


def silhouettes_downloader(size, **kwargs):
    if size not in (16, 28):
        raise ValueError("size must be 16 or 28")

    actual_filename = FILENAME.format(size)
    actual_url = BASE_URL + actual_filename
    default_downloader(urls=[actual_url],
                       filenames=[actual_filename], **kwargs)


def fill_subparser(subparser):
    """Sets up a subparser to download the Silhouettes dataset files.

    The following CalTech 101 Silhouette dataset files can be downloaded
    from Benjamin M. Marlin's website [MARLIN]:
    `caltech101_silhouettes_16_split1.mat` and
    `caltech101_silhouettes_28_split1.mat`.

    .. [MARLIN] https://people.cs.umass.edu/~marlin/data.shtml

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `caltech101_silhouettes` command.

    """
    subparser.add_argument(
        "size", type=int, choices=(16, 28),
        help="height/width of the datapoints")
    return silhouettes_downloader
