import socket
import os

from buffer import Buffer


def receive_file(HOST, PORT):
    # If server and client run in same local directory,
    # need a separate place to store the uploads.
    try:
        os.mkdir('RF_parameters')
    except FileExistsError:
        pass

    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("Waiting for a connection.....")

    while True:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = Buffer(conn)

        while True:
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            file_name = os.path.join('RF_parameters', file_name)
            print('file name: ', file_name)

            file_size = int(connbuf.get_utf8())
            print('file size: ', file_size)

            with open(file_name, 'wb') as f:
                remaining = file_size
                while remaining:
                    chunk_size = 4096 if remaining >= 4096 else remaining
                    chunk = connbuf.get_bytes(chunk_size)
                    if not chunk: break
                    f.write(chunk)
                    remaining -= len(chunk)
                if remaining:
                    print('File incomplete.  Missing', remaining, 'bytes.')
                else:
                    print('File received successfully.')
        print('Connection closed.')
        conn.close()


if __name__ == "__main__":
    HOST = '192.168.50.167'
    PORT = 65432
    receive_file(HOST, PORT)
