#!/usr/bin/env python
"""Fuel dataset downloading utility."""
import argparse
import os

from fuel import downloaders
from fuel.downloaders.base import NeedURLPrefix

url_prefix_message = """
Some files for this dataset do not have a download URL.

Provide a URL prefix with --url-prefix to prepend to the filenames,
e.g. http://path.to/files/
""".strip()


def main(args=None):
    """Entry point for `fuel-download` script.

    This function can also be imported and used from Python.

    Parameters
    ----------
    args : iterable, optional (default: None)
        A list of arguments that will be passed to Fuel's downloading
        utility. If this argument is not specified, `sys.argv[1:]` will
        be used.

    """
    built_in_datasets = dict(downloaders.all_downloaders)
    parser = argparse.ArgumentParser(
        description='Download script for built-in datasets.')
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-d", "--directory", help="where to save the downloaded files",
        type=str, default=os.getcwd())
    parent_parser.add_argument(
        "--clear", help="clear the downloaded files", action='store_true')
    subparsers = parser.add_subparsers()
    for name, subparser_fn in built_in_datasets.items():
        subparser_fn(subparsers.add_parser(
            name, parents=[parent_parser],
            help='Download the {} dataset'.format(name)))
    args = parser.parse_args()
    args_dict = vars(args)
    try:
        func = args_dict.pop('func')
    except KeyError:
        parser.print_usage()
        parser.exit()
    try:
        func(**args_dict)
    except NeedURLPrefix:
        parser.error(url_prefix_message)


if __name__ == "__main__":
    main()
