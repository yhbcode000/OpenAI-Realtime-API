from shared import *
from interface import Interface

class Client:
    def __init__(self, interface: Interface):
        self.interface = interface
