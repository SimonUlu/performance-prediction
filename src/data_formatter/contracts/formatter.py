from abc import ABC, abstractmethod

class Formatter(ABC):
    def __init__(self, file):
        super().__init__()
        self.file = file

    @abstractmethod 
    def format(self, data):
        pass