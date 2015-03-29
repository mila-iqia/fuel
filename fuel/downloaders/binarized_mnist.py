import os

from fuel.downloaders.base import download


def binarized_mnist(subparser):
    """Sets up a subparser to download the binarized MNIST dataset files.

    The binarized MNIST dataset files
    (`binarized_mnist_{train,valid,test}.amat`) are downloaded from
    Hugo Larochelle's website [HUGO].

    .. [HUGO] http://www.cs.toronto.edu/~larocheh/public/datasets/
       binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `binarized_mnist` command.

    """
    def download_binarized_mnist(args):
        save_directory = args.directory
        files = [os.path.join(save_directory,
                              'binarized_mnist_{}.amat'.format(s))
                 for s in ['train', 'valid', 'test']]
        urls = ['http://www.cs.toronto.edu/~larocheh/public/datasets/' +
                'binarized_mnist/binarized_mnist_{}.amat'.format(s)
                for s in ['train', 'valid', 'test']]
        if args.clear:
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)
        else:
            for url, f in zip(urls, files):
                download(url, f)

    subparser.add_argument("-d", "--directory", help="where to save the " +
                           "downloaded files", type=str, default=os.getcwd())
    subparser.add_argument("--clear", help="clear the downloaded files",
                           action='store_true')
    subparser.set_defaults(func=download_binarized_mnist)
