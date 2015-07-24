Installation
============

The easiest way to install Fuel is using the Python package manager ``pip``.
Fuel isn't listed yet on the Python Package Index (PyPI), so you will
have to grab it directly from GitHub.

.. code-block:: bash

   $ pip install git+git://github.com/mila-udem/fuel.git

This will give you the cutting-edge development version. The latest stable
release is in the ``stable`` branch and can be installed as follows.

.. code-block:: bash

   $ pip install git+git://github.com/mila-udem/fuel.git@stable

If you don't have administrative rights, add the ``--user`` switch to the
install commands to install the packages in your home folder. If you want to
update Fuel, simply repeat the first command with the ``--upgrade`` switch
added to pull the latest version from GitHub.

.. warning::

   Pip may try to install or update NumPy and SciPy if they are not present or
   outdated. However, pip's versions might not be linked to an optimized BLAS
   implementation. To prevent this from happening make sure you update NumPy
   and SciPy using your system's package manager (e.g.  ``apt-get`` or
   ``yum``), or use a Python distribution like Anaconda_, before installing
   Fuel. You can also pass the ``--no-deps`` switch and install all the
   requirements manually.

   If the installation crashes with ``ImportError: No module named
   numpy.distutils.core``, install NumPy and try again again.


Requirements
------------
Fuel's requirements are

* PyYAML_, to parse the configuration file
* six_, to support both Python 2 and 3 with a single codebase
* h5py_ and PyTables_ for the HDF5 storage back-end
* pillow_, providing PIL for image preprocessing
* Cython_, for fast extensions
* pyzmq_, to efficiently send data across processes
* picklable_itertools_, for supporting iterator serialization
* SciPy_, to read from MATLAB's .mat format
* requests_, to download canonical datasets

nose2_ is an optional requirement, used to run the tests.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _nose2: https://nose2.readthedocs.org/
.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _six: http://pythonhosted.org/six/
.. _h5py: http://www.h5py.org/
.. _PyTables: http://www.pytables.org/
.. _SciPy: http://www.scipy.org/
.. _pillow: https://python-pillow.github.io/
.. _Cython: http://cython.org/
.. _pyzmq: https://zeromq.github.io/pyzmq/
.. _picklable_itertools: https://github.com/dwf/picklable_itertools
.. _requests: http://docs.python-requests.org/en/latest/

Development
-----------

If you want to work on Fuel's development, your first step is to `fork Fuel
on GitHub`_. You will now want to install your fork of Fuel in editable mode.
To install in your home directory, use the following command, replacing ``USER``
with your own GitHub user name:

.. code-block:: bash

   $ pip install -e git+git@github.com:USER/fuel.git#egg=fuel[test,docs] --src=$HOME

As with the usual installation, you can use ``--user`` or ``--no-deps`` if you
need to. You can now make changes in the ``fuel`` directory created by pip,
push to your repository and make a pull request.

If you had already cloned the GitHub repository, you can use the following
command from the folder you cloned Fuel to:

.. code-block:: bash

   $ pip install -e file:.#egg=fuel[test,docs]

Fuel contains Cython extensions, which need to be recompiled if you
update the Cython `.pyx` files. Each time these files are modified, you
should run:

.. code-block:: bash

   $ python setup.py build_ext --inplace

.. _fork Fuel on GitHub: https://github.com/mila-udem/fuel/fork

Documentation
~~~~~~~~~~~~~

If you want to build a local copy of the documentation, you can follow
the instructions in the `documentation development guidelines`_.

.. _documentation development guidelines:
   http://blocks.readthedocs.org/en/latest/development/docs.html
