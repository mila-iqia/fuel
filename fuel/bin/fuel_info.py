#!/usr/bin/env python
"""Fuel utility for extracting metadata."""
import argparse
import os

import h5py

message_prefix_template = 'Metadata for {}'
message_body_template = """

    The command used to generate this file is

        {}

    Relevant versions are

        H5PYDataset     {}
        fuel.converters {}
"""


def main(args=None):
    """Entry point for `fuel-info` script.

    This function can also be imported and used from Python.

    Parameters
    ----------
    args : iterable, optional (default: None)
        A list of arguments that will be passed to Fuel's information
        utility. If this argument is not specified, `sys.argv[1:]` will
        be used.

    """
    parser = argparse.ArgumentParser(
        description='Extracts metadata from a Fuel-converted HDF5 file.')
    parser.add_argument("filename", help="HDF5 file to analyze")
    args = parser.parse_args()

    with h5py.File(args.filename, 'r') as h5file:
        interface_version = h5file.attrs.get('h5py_interface_version', 'N/A')
        fuel_convert_version = h5file.attrs.get('fuel_convert_version', 'N/A')
        fuel_convert_command = h5file.attrs.get('fuel_convert_command', 'N/A')

    message_prefix = message_prefix_template.format(
        os.path.basename(args.filename))
    message_body = message_body_template.format(
        fuel_convert_command, interface_version, fuel_convert_version)
    message = ''.join(['\n', message_prefix, '\n', '=' * len(message_prefix),
                       message_body])
    print(message)


if __name__ == "__main__":
    main()
