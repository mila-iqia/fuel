import logging

import numpy
import zmq
from numpy.lib.format import header_data_from_array_1_0

from fuel.utils import buffer_

logger = logging.getLogger(__name__)


def send_arrays(socket, arrays, stop=False):
    """Send NumPy arrays using the buffer interface and some metadata.

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

    Notes
    -----
    The protocol is very simple: A single JSON object describing the array
    format (using the same specification as ``.npy`` files) is sent first.
    Subsequently the arrays are sent as bytestreams (through NumPy's
    support of the buffering protocol).

    """
    if arrays:
        # The buffer protocol only works on contiguous arrays
        arrays = [numpy.ascontiguousarray(array) for array in arrays]
    if stop:
        headers = {'stop': True}
        socket.send_json(headers)
    else:
        headers = [header_data_from_array_1_0(array) for array in arrays]
        socket.send_json(headers, zmq.SNDMORE)
        for array in arrays[:-1]:
            socket.send(array, zmq.SNDMORE)
        socket.send(arrays[-1])


def recv_arrays(socket):
    """Receive a list of NumPy arrays.

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
        If the first JSON object received contains the key `stop`,
        signifying that the server has finished a single epoch.

    """
    headers = socket.recv_json()
    if 'stop' in headers:
        raise StopIteration
    arrays = []
    for header in headers:
        data = socket.recv()
        buf = buffer_(data)
        array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
        array.shape = header['shape']
        if header['fortran_order']:
            array.shape = header['shape'][::-1]
            array = array.transpose()
        arrays.append(array)
    return arrays


def start_server(data_stream, port=5557, hwm=10):
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
    port : int, optional
        The port the server and the client (training loop) will use to
        communicate. Defaults to 5557.
    hwm : int, optional
        The `ZeroMQ high-water mark (HWM)
        <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
        sending socket. Increasing this increases the buffer, which can be
        useful if your data preprocessing times are very random.  However,
        it will increase memory usage. There is no easy way to tell how
        many batches will actually be queued with a particular HWM.
        Defaults to 10. Be sure to set the corresponding HWM on the
        receiving end as well.

    """
    logging.basicConfig(level='INFO')

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.set_hwm(hwm)
    socket.bind('tcp://*:{}'.format(port))

    it = data_stream.get_epoch_iterator()

    logger.info('server started')
    while True:
        try:
            data = next(it)
            stop = False
            logger.debug("sending {} arrays".format(len(data)))
        except StopIteration:
            it = data_stream.get_epoch_iterator()
            data = None
            stop = True
            logger.debug("sending StopIteration")
        send_arrays(socket, data, stop=stop)
