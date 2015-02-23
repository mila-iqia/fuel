"""This is an example of how to ask the server to buffer/send data.

The interface needs to be wrapped in a user-friendly ServerDataStream
class.

"""
import logging

import six
import zmq

from fuel.server import recv_array

BUFFER_SIZE = 5
LOGGER = logging.getLogger('client')
logging.basicConfig(level='INFO')

# Prepare our context and sockets
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5559")


# Let the buffer fill
while True:
    LOGGER.info('Sending buffering request')
    socket.send(b"buffer")
    message, = socket.recv_multipart()
    buffer_size = six.byte2int(message[0])
    LOGGER.info('Broker has buffered {} examples'.format(buffer_size))
    if buffer_size >= BUFFER_SIZE:
        LOGGER.info('Broker buffer is full.')
        break

# Request 10 batches
for _ in range(10):
    LOGGER.info("Requesting NumPy array")
    socket.send(b"next")
    data = recv_array(socket)
    LOGGER.info("Received NumPy array: {}...".format(str(data).split('\n')[0]))
