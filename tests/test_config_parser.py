import os
import tempfile

from numpy.testing import assert_equal, assert_raises

from fuel.config_parser import (Configuration, ConfigurationError,
                                extra_downloader_converter)


class TestExtraDownloaderConverter(object):
    def test_iterable(self):
        assert_equal(extra_downloader_converter(['a.b.c', 'd.e.f']),
                     ['a.b.c', 'd.e.f'])

    def test_str(self):
        assert_equal(extra_downloader_converter("a.b.c d.e.f"),
                     ['a.b.c', 'd.e.f'])

    def test_str_one_element(self):
        assert_equal(extra_downloader_converter("a.b.c"), ['a.b.c'])


def test_config_parser():
    _environ = dict(os.environ)
    try:

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('data_path: yaml_path')
            filename = f.name
        os.environ['FUEL_CONFIG'] = filename
        if 'FUEL_DATA_PATH' in os.environ:
            del os.environ['FUEL_DATA_PATH']
        config = Configuration()
        config.add_config('data_path', str, env_var='FUEL_DATA_PATH')
        config.add_config('config_with_default', int, default='1',
                          env_var='FUEL_CONFIG_TEST')
        config.add_config('config_without_default', str)
        config.load_yaml()
        assert config.data_path == 'yaml_path'
        os.environ['FUEL_DATA_PATH'] = 'env_path'
        assert config.data_path == 'env_path'
        assert config.config_with_default == 1
        os.environ['FUEL_CONFIG_TEST'] = '2'
        assert config.config_with_default == 2
        assert_raises(AttributeError, getattr, config,
                      'non_existing_config')
        assert_raises(ConfigurationError, getattr, config,
                      'config_without_default')
        config.data_path = 'manual_path'
        assert config.data_path == 'manual_path'
        config.new_config = 'new_config'
        assert config.new_config == 'new_config'
    finally:
        os.environ.clear()
        os.environ.update(_environ)
