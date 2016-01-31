from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the CelebA dataset file.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `celeba` command.

    """
    urls = ['https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
            'AAB7G69NLjRNqv_tyiULHSVUa/list_attr_celeba.txt?dl=1',
            'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
            'AADVdnYbokd7TXhpvfWLL3sga/img_align_celeba.zip?dl=1']
    filenames = ['list_attr_celeba.txt', 'img_align_celeba.zip']
    subparser.set_defaults(urls=urls, filenames=filenames)
    return default_downloader
