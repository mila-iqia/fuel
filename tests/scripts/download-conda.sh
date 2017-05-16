#!/usr/bin/sh
if [[ $TRAVIS_PYTHON_VERSION == 2.7 ]]; then
   wget -q http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
else
   wget -q http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
chmod +x miniconda.sh
