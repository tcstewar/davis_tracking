import socket
import struct

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('',21212))

format = '<HH'
packet_size = struct.calcsize(format)
print(packet_size)

while True:
    data = s.recv(packet_size)
    result = struct.unpack(format, data)
    print(result)
