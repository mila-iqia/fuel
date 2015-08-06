"""Installation script."""
from os import path
import sys
from setuptools import find_packages, setup
from Cython.Build import cythonize
from distutils.extension import Extension

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.rst')) as f:
    LONG_DESCRIPTION = f.read().strip()

# Visual C++ apparently doesn't respect/know what to do with this flag.
# Windows users may thus see unused function warnings. Oh well.
if sys.platform != 'win32':
    extra_compile_args = ['-Wno-unused-function']
else:
    extra_compile_args = []

setup(
    name='fuel',
    version='0.0.1',  # PEP 440 compliant
    description='Data pipeline framework for machine learning',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/mila-udem/fuel.git',
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
    install_requires=['six', 'picklable_itertools', 'pyyaml', 'h5py', 'cython',
                      'tables', 'progressbar2', 'pyzmq', 'scipy', 'pillow',
                      'requests'],
    extras_require={
        'test': ['nose', 'nose2', 'mock']
    },
    entry_points={
        'console_scripts': ['fuel-convert = fuel.bin.fuel_convert:main',
                            'fuel-download = fuel.bin.fuel_download:main',
                            'fuel-info = fuel.bin.fuel_info:main']
    },
    ext_modules=cythonize(Extension("fuel.transformers._image",
                                    ["fuel/transformers/_image.pyx"],
                                    extra_compile_args=extra_compile_args))
)
