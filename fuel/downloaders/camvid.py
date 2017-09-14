from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the Camvid dataset file.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `camvid` command.
    """
    url = ['To be definied']
    filenames = ['camvid_dataset.zip']
    subparser.set_defaults(urls=url, filenames=filenames)
    return default_downloader
