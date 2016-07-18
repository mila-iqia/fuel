from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Set up a subparser to download the MJSynth dataset file.

    The MJSynth dataset file `mjsynth.tar.gz` is downloaded from the VGG
    Text Recognition Data webpage [VGG].

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mjsynth` command.

    """
    subparser.set_defaults(
        urls=['http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz'],
        filenames=['mjsynth.tar.gz'])
    return default_downloader
