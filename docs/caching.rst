Caching datasets locally
========================

In some use cases, it may be desirable to set Fuel's ``data_path`` to
point to a shared network drive. For example, when configuring multiple
machines in a cluster to work on the same data in parallel.
However, this can easily cause network bandwidth to become saturated.

To avoid this problem, Fuel provides a second configuration variable
named ``local_data_path``, which can be set in ``~/.fuelrc``. This
variable points to a filesystem directory to be used to act as a local
cache for datasets.

This variable can also be set through an environment variable as follows:

.. code-block:: bash

    $ export FUEL_LOCAL_DATA_PATH="/home/username/my_local_cache"
