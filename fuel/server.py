import logging
from collections import deque
from multiprocessing import Process

import numpy
import six
import zmq

from fuel.utils import buffer_

LOGGER = logging.getLogger('server')
logging.basicConfig(level='INFO')


def send_arrays(socket, arrays, flags=0, copy=True, track=False, stop=False):
    """Send a NumPy array using the buffer interface and some metadata."""
    if stop:
        mds = {'stop': True}
        socket.send_json(mds, flags)
    else:
        mds = [{'dtype': str(array.dtype), 'shape': array.shape}
               for array in arrays]
        socket.send_json(mds, flags | zmq.SNDMORE)
        for array in arrays[:-1]:
            socket.send(array, flags | zmq.SNDMORE, copy=copy, track=track)
        return socket.send(arrays[-1], flags, copy=copy, track=track)


def recv_arrays(socket, flags=0, copy=True, track=False):
    """Receive a NumPy array."""
    mds = socket.recv_json(flags=flags)
    if 'stop' in mds:
        raise StopIteration
    arrays = []
    for md in mds:
        data = socket.recv(flags=flags, copy=copy, track=track)
        buf = buffer_(data)
        array = numpy.frombuffer(buf, dtype=md['dtype'])
        arrays.append(array.reshape(md['shape']))
    return arrays


def server(data_stream):
    """The main process that processes and sends data."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:5560")

    it = data_stream.get_epoch_iterator()

    while True:
        socket.recv()
        LOGGER.info("Sending NumPy array.")
        try:
            data = next(it)
            stop = False
        except StopIteration:
            it = data_stream.get_epoch_iterator()
            data = None
            stop = True
        send_arrays(socket, data, stop=stop)


def broker():
    """The broker is run in a separate process and holds the queue."""
    # Prepare our context and sockets
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.DEALER)
    frontend.bind("tcp://*:5559")
    backend.bind("tcp://*:5560")

    # Initialize poll set
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    # Create queue
    queue = deque()
    to_buffer = 0

    # Switch messages between sockets
    while True:
        socks = dict(poller.poll())

        if socks.get(frontend) == zmq.POLLIN:
            message = frontend.recv_multipart()
            if message[2] == six.b("buffer"):
                LOGGER.info("Received buffering request.")
                to_buffer += 1
            else:
                LOGGER.info("Sending data to client.")
                frontend.send_multipart(queue.popleft())
            backend.send_multipart(message)

        if socks.get(backend) == zmq.POLLIN:
            message = backend.recv_multipart()
            queue.append(message)
            if to_buffer:
                LOGGER.info("Buffering to {}.".format(len(queue)))
                frontend.send_multipart(message[:2] +
                                        [six.int2byte(len(queue))])
                to_buffer -= 1


def start_server(data_stream):
    """Start a data processing server.

    Parameters
    ----------
    data_stream : :class:`.DataStream`
        The data stream to return examples from.

    """
    Process(target=broker).start()
    server(data_stream)

if __name__ == "__main__":
    # A sample server that returns MNIST batches
    from fuel.datasets import MNIST
    from fuel.streams import DataStream
    from fuel.schemes import SequentialScheme
    mnist = MNIST('train')
    data_stream = DataStream(
        mnist, iteration_scheme=SequentialScheme(1500, 500)
    )
    start_server(data_stream)
