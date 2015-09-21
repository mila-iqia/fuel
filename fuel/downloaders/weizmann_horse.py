from fuel.downloaders.base import default_downloader


def fill_subparser(subparser):
    """Sets up a subparser to download the Weizmann Horse Dataset files.

    The following Weizmann Horse Dataset file is downloaded from Eran
    Borenstein's website [BORENSTEIN]:
    `weizmann_horse_db.tar.gz`

    .. [BORENSTEIN] http://www.msri.org/people/members/eranb/

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `weizmann_horses` command.

    """
    filenames = ['weizmann_horse_db.tar.gz']
    urls = ['http://www.msri.org/people/members/eranb/' + f for f in filenames]
    subparser.set_defaults(urls=urls, filenames=filenames)
    return default_downloader
