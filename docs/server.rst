Parallelizing data processing
=============================

Here's a scenario that is commonly encountered in practice: a big model is
trained on a large dataset that doesn't fit in memory (e.g. a deep convolutional
neural network trained on ImageNet) using a GPU to accelerate training.

In that case, doing data processing and training in a single process is very
inefficient: the GPU is idle when data is read off disk and processed, and
nothing else is done while the GPU is at work.

An obvious solution is to do the preprocessing and training in parallel: if I/O
operations are executed while the GPU is busy, then less time is wasted waiting
for data to be available.

In this section, we'll cover how to spawn a data processing server in a separate
process and how to connect to that server and consume that data in a training
script.

Toy example
-----------

Let's first create a dummy dataset:

>>> from fuel.datasets import IndexableDataset
>>> dataset = IndexableDataset({'features': [[0] * 128] * 1000})

In practice, the dataset can be whatever you want, but we'll settle with that
for simplicity.

Since this is a pretty small dataset, we'll need to simulate slowdowns
associated with I/O operations and preprocessing. We'll create a transformer
whose sole purpose is to wait some period of time before returning the requested
data:

>>> import time
>>> from fuel.transformers import Transformer
>>> class Bottleneck(Transformer):
...     def __init__(self, *args, **kwargs):
...         self.slowdown = kwargs.pop('slowdown', 0)
...         super(Bottleneck, self).__init__(*args, **kwargs)
...
...     def get_data(self, request=None):
...         if request is not None:
...             raise ValueError
...         time.sleep(self.slowdown)
...         return next(self.child_epoch_iterator)

We'll also create a context manager to time a block of code and print the
result:

>>> from contextlib import contextmanager
>>> @contextmanager
... def timer(name):
...     start_time = time.time()
...     yield
...     stop_time = time.time()
...     print('{} took {} seconds'.format(name, stop_time - start_time))

Let's see how much of a slowdown we're incurring in our current setup:

>>> from fuel.schemes import ShuffledScheme
>>> from fuel.streams import DataStream
>>> iteration_scheme = ShuffledScheme(examples=1000, batch_size=100)
>>> data_stream = Bottleneck(
...     data_stream=DataStream.default_stream(
...         dataset=dataset, iteration_scheme=iteration_scheme),
...     slowdown=0.005)
>>> with timer('Iteration'): # doctest: +ELLIPSIS
...     for data in data_stream.get_epoch_iterator(): pass
Iteration took ... seconds

Next, we'll write a small piece of code that simulates some computation being
done on our data. Let's pretend that we're training for 5 epochs and that
training on a batch takes a fixed amount of time.

>>> with timer('Training'): # doctest: +ELLIPSIS
...     for i in range(5):
...         for data in data_stream.get_epoch_iterator(): time.sleep(0.01)
Training took ... seconds

Take note of the time it takes: we'll cut that down with the data processing
server.

Data processing server
----------------------

Fuel features a :func:`~.server.start_server` built-in function which takes a
data stream as input and sets up a data processing server that iterates over
this data stream. The function signature is the following:

.. code-block:: python

    def start_server(data_stream, port=5557, hwm=10):

The ``data_stream`` argument is self-explanatory. The port the server listens to
defaults to 5557 but can be changed through the ``port`` argument. The ``hwm``
argument controls the high-water mark, which loosely translates to the buffer
size. The default value of 10 usually works well. Increasing this increases the
buffer, which can be useful if your data preprocessing times are very random.
However, it will increase memory usage. Be sure to set the corresponding
high-water mark on the client as well.

A client can then connect to that server and request data. The
:class:`~.streams.ServerDataStream` class is what we want to use. Its
``__init__`` method has the following signature:

.. code-block:: python

    def __init__(self, sources, host='localhost', port=5557, hwm=10):

The ``sources`` argument is how you communicate source names to the data stream.
It's expected to be a tuple of strings with as many elements as there are
sources that will be received. The ``host`` and ``port`` arguments are used to
specify where to connect to the data processing server. Note that this allows
you to run the server on a completely different machine! The ``hwm`` argument
should mirror what you passed to :func:`start_server`.

Putting it together
-------------------

You'll need to separate your code in two files: one that spawns a data
processing server and one that handles the training loop.

Here's those two files:

.. code-block:: python

    """server.py"""
    import time

    from fuel.datasets import IndexableDataset
    from fuel.schemes import ShuffledScheme
    from fuel.server import start_server
    from fuel.streams import DataStream
    from fuel.transformers import Transformer


    class Bottleneck(Transformer):
        """Waits every time data is requested to simulate a bottleneck.

        Parameters
        ----------
        slowdown : float, optional
            Time (in seconds) to wait before returning data. Defaults to 0.

        """
        def __init__(self, *args, **kwargs):
            self.slowdown = kwargs.pop('slowdown', 0)
            super(Bottleneck, self).__init__(*args, **kwargs)

        def get_data(self, request=None):
            if request is not None:
                raise ValueError
            time.sleep(self.slowdown)
            return next(self.child_epoch_iterator)


    def create_data_stream(slowdown=0):
        """Creates a bottlenecked data stream of dummy data.

        Parameters
        ----------
        slowdown : float
            Time (in seconds) to wait each time data is requested.

        Returns
        -------
        data_stream : fuel.streams.AbstactDataStream
            Bottlenecked data stream.

        """
        dataset = IndexableDataset({'features': [[0] * 128] * 1000})
        iteration_scheme = ShuffledScheme(examples=1000, batch_size=100)
        data_stream = Bottleneck(
            data_stream=DataStream.default_stream(
                dataset=dataset, iteration_scheme=iteration_scheme),
            slowdown=slowdown)
        return data_stream


    if __name__ == "__main__":
        start_server(create_data_stream(0.005))


.. code-block:: python

    """train.py"""
    import argparse
    import time
    from contextlib import contextmanager

    from fuel.streams import ServerDataStream

    from server import create_data_stream


    @contextmanager
    def timer(name):
        """Times a block of code and prints the result.

        Parameters
        ----------
        name : str
            What this block of code represents.

        """
        start_time = time.time()
        yield
        stop_time = time.time()
        print('{} took {} seconds'.format(name, stop_time - start_time))


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-p', '--parallel', action='store_true',
            help='run data preprocessing in a separate process')
        args = parser.parse_args()

        if args.parallel:
            data_stream = ServerDataStream(('features',))
        else:
            data_stream = create_data_stream(0.005)

        with timer('Training'):
            for i in range(5):
                for data in data_stream.get_epoch_iterator(): time.sleep(0.01)

We've modularized the code to be a little more convenient to re-use. Save the
two files in the same directory and type

.. code-block:: bash

    $ python train.py    

This will run the training and the data processing in the same process.

Now, type

.. code-block:: bash

    $ python server.py

in a separate terminal window and type

.. code-block:: bash

    $ python train.py -p

Compare the two running times: you should see a clear gain using the
separate data processing server.
