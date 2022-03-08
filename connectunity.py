from http import client
import os
import socket

class connectunity:
    def __init__(self, host, mainport, is_connect):
        self.host = host
        self.mainport = mainport
        self.is_connect = is_connect

    def ConnectUnity(self):
        if (self.is_connect == False):
            return
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client.connect((self.host, self.mainport))

    def Send(self, data):
        if (self.is_connect == False):
            return
        # data.encode('utf-8')
        self.client.send(data)

    def Receive(self, sec):
        if (self.is_connect == False):
            return

        data = self.client.recv(sec)
        print(data.decode('utf-8'))
        return data