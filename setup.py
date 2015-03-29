"""Installation script."""
from os import path
from setuptools import find_packages, setup

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.rst')) as f:
    LONG_DESCRIPTION = f.read().strip()

setup(
    name='fuel',
    version='0.1a1',  # PEP 440 compliant
    description='Data pipeline framework for machine learning',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/bartvm/fuel.git',
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
    install_requires=['six', 'picklable_itertools', 'toolz', 'pyyaml', 'h5py',
                      'tables', 'urllib3', 'certifi'],
    scripts=['bin/fuel-convert', 'bin/fuel-download']
)
