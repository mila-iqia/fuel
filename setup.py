"""Installation script."""
from os import path
import sys
from io import open
from setuptools import find_packages, setup
from distutils.extension import Extension

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read().strip()

# Visual C++ apparently doesn't respect/know what to do with this flag.
# Windows users may thus see unused function warnings. Oh well.
if sys.platform != 'win32':
    extra_compile_args = ['-Wno-unused-function']
else:
    extra_compile_args = []

exec_results = {}
with open(path.join(path.dirname(__file__), 'fuel/version.py')) as file_:
    exec(file_.read(), exec_results)
version = exec_results['version']

setup(
    name='fuel',
    version=version,  # PEP 440 compliant
    description='Data pipeline framework for machine learning',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/mila-udem/fuel.git',
    download_url='https://github.com/mila-udem/fuel/tarball/v' + version,
    author='Universite de Montreal',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='dataset data iteration pipeline processing',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'six', 'picklable_itertools', 'pyyaml', 'h5py', 'tables',
                      'progressbar2', 'pyzmq', 'scipy', 'pillow', 'requests'],
    extras_require={
        'test': ['mock', 'nose', 'nose2'],
        'docs': ['sphinx', 'sphinx-rtd-theme']
    },
    entry_points={
        'console_scripts': ['fuel-convert = fuel.bin.fuel_convert:main',
                            'fuel-download = fuel.bin.fuel_download:main',
                            'fuel-info = fuel.bin.fuel_info:main']
    },
    ext_modules=[Extension("fuel.transformers._image",
                           ["fuel/transformers/_image.c"],
                           extra_compile_args=extra_compile_args)]
)
