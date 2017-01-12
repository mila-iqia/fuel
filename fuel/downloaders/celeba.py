from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the CelebA dataset file.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `celeba` command.

    """
    urls = ['https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
            'AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1',
            'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
            'AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1']
    filenames = ['list_attr_celeba.txt', 'img_align_celeba.zip']
    subparser.set_defaults(urls=urls, filenames=filenames)
    return default_downloader
