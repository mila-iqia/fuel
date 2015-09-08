from importlib import import_module
from unittest.case import SkipTest

from fuel.utils import find_in_data_path
from fuel import config


def skip_if_not_available(modules=None, datasets=None, configurations=None):
    """Raises a SkipTest exception when requirements are not met.

    Parameters
    ----------
    modules : list
        A list of strings of module names. If one of the modules fails to
        import, the test will be skipped.
    datasets : list
        A list of strings of folder names. If the data path is not
        configured, or the folder does not exist, the test is skipped.
    configurations : list
        A list of of strings of configuration names. If this configuration
        is not set and does not have a default, the test will be skipped.

    """
    if modules is None:
        modules = []
    if datasets is None:
        datasets = []
    if configurations is None:
        configurations = []
    for module in modules:
        try:
            import_module(module)
        except Exception:
            raise SkipTest
    if datasets and not hasattr(config, 'data_path'):
        raise SkipTest
    for dataset in datasets:
        try:
            find_in_data_path(dataset)
        except IOError:
            raise SkipTest
    for configuration in configurations:
        if not hasattr(config, configuration):
            raise SkipTest
