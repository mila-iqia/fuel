import logging
from collections import deque
from multiprocessing import Process

import numpy
import six
import zmq

from fuel.utils import buffer_

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


def send_arrays(socket, arrays, flags=0, copy=True, track=False, stop=False):
    """Send a NumPy array using the buffer interface and some metadata.

    Parameters
    ----------
    socket : :class:`zmq.Socket`
        The socket to send data over.
    arrays : list
        A list of :class:`numpy.ndarray` to transfer.
    stop : bool, optional
        Instead of sending a series of NumPy arrays, send a JSON object
        with a single `stop` key. The :func:`recv_arrays` will raise
        ``StopIteration`` when it receives this.

    See Also
    --------
    :meth:`zmq.Socket.send` for other arguments

    Notes
    -----
    The protocol is very simple: A single JSON object that contains the
    shape and data type for each array is transferred first. Subsequently
    the arrays are sent as bytestreams (through NumPy's support of the
    buffering protocol).

    """
    if stop:
        mds = {'stop': True}
        return socket.send_json(mds, flags)
    else:
        mds = [{'dtype': str(array.dtype), 'shape': array.shape}
               for array in arrays]
        socket.send_json(mds, flags | zmq.SNDMORE)
        for array in arrays[:-1]:
            socket.send(array, flags | zmq.SNDMORE, copy=copy, track=track)
        return socket.send(arrays[-1], flags, copy=copy, track=track)


def recv_arrays(socket, flags=0, copy=True, track=False):
    """Receive a NumPy array.

    Parameters
    ----------
    socket : :class:`zmq.Socket`
        The socket to receive the arrays on.

    Returns
    -------
    list
        A list of :class:`numpy.ndarray` objects.

    Raises
    ------
    StopIteration
        If the first JSON object received contains the key `stop`.

    See Also
    --------
    :func:`send_arrays` for an explanation of the protocol used to transfer
    arrays, and :meth:`zmq.Socket.recv` for an explanation of the other
    arguments.

    """
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


def server(data_stream, server_port):
    """The main process that processes and sends data."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:{}".format(server_port))

    it = data_stream.get_epoch_iterator()

    logger.info('server started')
    while True:
        socket.recv()
        try:
            data = next(it)
            stop = False
            logger.info("sending {} arrays".format(len(data)))
        except StopIteration:
            it = data_stream.get_epoch_iterator()
            data = None
            stop = True
            logger.info("sending StopIteration")
        send_arrays(socket, data, stop=stop)


def broker(server_port, client_port):
    """The broker is run in a separate process and holds the queue."""
    logger.debug('binding broker to sockets')
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.DEALER)
    frontend.bind("tcp://*:{}".format(client_port))
    backend.bind("tcp://*:{}".format(server_port))

    logger.debug('broker starting to poll')
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    # Create queue
    queue = deque()
    to_buffer = 0

    logger.info('broker started')
    while True:
        socks = dict(poller.poll())

        if socks.get(frontend) == zmq.POLLIN:
            message = frontend.recv_multipart()
            if message[2] == six.b("buffer"):
                logger.debug("broker received request to buffer")
                to_buffer += 1
            else:
                logger.debug("broker responding to data request from client")
                frontend.send_multipart(queue.popleft())
            logger.debug("broker requesting data from server")
            backend.send_multipart(message)

        if socks.get(backend) == zmq.POLLIN:
            logger.debug("broker receiving data from server")
            message = backend.recv_multipart()
            queue.append(message)
            if to_buffer:
                logger.debug("broker buffering data")
                frontend.send_multipart(message[:2] +
                                        [six.int2byte(len(queue))])
                to_buffer -= 1


def start_server(data_stream, server_port=5560, client_port=5559):
    """Start a data processing server.

    This command starts a server in the current process that performs the
    actual data processing (by retrieving data from the given data stream).
    It also starts a second process, the broker, which mediates between the
    server and the client. The broker also keeps a buffer of batches in
    memory.

    Parameters
    ----------
    data_stream : :class:`.DataStream`
        The data stream to return examples from.
    server_port : int
        The port the server and the broker will use to communicate.
    client_port : int
        The port that the broker will communicate with the client on.

    """
    Process(target=broker, args=(server_port, client_port)).start()
    server(data_stream, server_port)

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
