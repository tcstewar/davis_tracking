import socket
import struct
import nengo

class UDPReceiver(object):
    def __init__(self, port, size_out):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('', port))
        self.port = port
        self.socket.setblocking(0)
        self.format = '<' + 'H'*size_out
        self.packet_size = struct.calcsize(self.format)
        self.data = [0] * size_out
    def make_node(self):
        def receive(t):
            try:
                while True:  # empty the buffer
                    data = self.socket.recv(self.packet_size)
                    result = struct.unpack(self.format, data)
                    self.data[:] = result
            except socket.error:
                pass
            return self.data
        return nengo.Node(receive, size_in=0, size_out=2)

model = nengo.Network()
with model:

    pos = UDPReceiver(21212, size_out=2).make_node()

