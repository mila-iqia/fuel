from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the Dogs vs. Cats dataset file.

    Kaggle's Dogs vs. Cats [KAGGLE] dataset is downloaded from Dropbox
    since Kaggle requires user authentication.

    .. [KAGGLE] https://www.kaggle.com/c/dogs-vs-cats

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `dogs_vs_cats` command.

    """
    urls = ['https://www.dropbox.com/s/s3u30quvpxqdbz6/train.zip?dl=1',
            'https://www.dropbox.com/s/21rwu6drnplsbkb/test1.zip?dl=1']
    filenames = ['dogs_vs_cats.train.zip', 'dogs_vs_cats.test1.zip']
    subparser.set_defaults(urls=urls, filenames=filenames)
    return default_downloader
