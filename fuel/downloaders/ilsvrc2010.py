from fuel.converters.ilsvrc2010 import IMAGE_TARS
from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the ILSVRC2010 dataset files.

    Note that you will need to use `--url-prefix` to download the
    non-public files (namely, the TARs of images). This is a single
    prefix that is common to all distributed files, which you can
    obtain by registering at the ImageNet website [DOWNLOAD].

    Note that these files are quite large and you may be better off
    simply downloading them separately and running ``fuel-convert``.

    .. [DOWNLOAD] http://www.image-net.org/download-images


    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `ilsvrc2010` command.

    """
    urls = [
        ('http://www.image-net.org/challenges/LSVRC/2010/'
         'ILSVRC2010_test_ground_truth.txt'),
        ('http://www.image-net.org/challenges/LSVRC/2010/'
         'download/ILSVRC2010_devkit-1.0.tar.gz'),
    ] + ([None] * len(IMAGE_TARS))
    filenames = [None, None] + list(IMAGE_TARS)
    subparser.set_defaults(urls=urls, filenames=filenames)
    subparser.add_argument('-P', '--url-prefix', type=str, default=None,
                           help="URL prefix to prepend to the filenames of "
                                "non-public files, in order to download them. "
                                "Be sure to include the trailing slash.")
    return default_downloader
