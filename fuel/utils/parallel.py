"""Utilities for speeding things up through parallelism.

Currently including:

* A very simple PUSH-PULL reusable producer-consumer pattern
  using a ZeroMQ socket instead of the (slow, unnecessarily
  copying) multiprocessing.Queue. See :func:`producer_consumer`.

"""
from multiprocessing import Process
import zmq


def _producer_wrapper(f, port, addr='tcp://127.0.0.1'):
    """A shim that sets up a socket and starts the producer callable.

    Parameters
    ----------
    f : callable
        Callable that takes a single argument, a handle
        for a ZeroMQ PUSH socket. Must be picklable.
    port : int
        The port on which the socket should connect.
    addr : str, optional
        Address to which the socket should connect. Defaults
        to localhost ('tcp://127.0.0.1').

    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect(':'.join([addr, str(port)]))
        f(socket)
    finally:
        # Works around a Python 3.x bug.
        context.destroy()


def _spawn_producer(f, port, addr='tcp://127.0.0.1'):
    """Start a process that sends results on a PUSH socket.

    Parameters
    ----------
    f : callable
        Callable that takes a single argument, a handle
        for a ZeroMQ PUSH socket. Must be picklable.

    Returns
    -------
    process : multiprocessing.Process
        The process handle of the created producer process.

    """
    process = Process(target=_producer_wrapper, args=(f, port, addr))
    process.start()
    return process


def producer_consumer(producer, consumer, addr='tcp://127.0.0.1',
                      port=None, context=None):
    """A producer-consumer pattern.

    Parameters
    ----------
    producer : callable
        Callable that takes a single argument, a handle
        for a ZeroMQ PUSH socket. Must be picklable.
    consumer : callable
        Callable that takes a single argument, a handle
        for a ZeroMQ PULL socket.
    addr : str, optional
        Address to which the socket should connect. Defaults
        to localhost ('tcp://127.0.0.1').
    port : int, optional
        The port on which the consumer should listen.
    context : zmq.Context, optional
        The ZeroMQ Context to use. One will be created otherwise.

    Returns
    -------
    result
        Passes along whatever `consumer` returns.

    Notes
    -----
    This sets up a PULL socket in the calling process and forks
    a process that calls `producer` on a PUSH socket. When the
    consumer returns, the producer process is terminated.

    Wrap `consumer` or `producer` in a `functools.partial` object
    in order to send additional arguments; the callables passed in
    should expect only one required, positional argument, the socket
    handle.

    """
    context_created = False
    if context is None:
        context_created = True
        context = zmq.Context()
    try:
        consumer_socket = context.socket(zmq.PULL)
        if port is None:
            port = consumer_socket.bind_to_random_port(addr)
        try:
            process = _spawn_producer(producer, port)
            result = consumer(consumer_socket)
        finally:
            process.terminate()
        return result
    finally:
        # Works around a Python 3.x bug.
        if context_created:
            context.destroy()
