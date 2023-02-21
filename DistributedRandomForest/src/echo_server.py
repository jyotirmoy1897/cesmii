import socket
import numpy as np
import pickle

HOST = '192.168.50.167'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
data = b""
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        #shape_packet = conn.recv(4096)
        #shape = shape_packet.decode("utf-8")
        print(pickle.loads(data))
        #print(np.frombuffer(data))
        #print(shape)
