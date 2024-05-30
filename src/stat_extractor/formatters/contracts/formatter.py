from abc import ABC, abstractmethod
import pandas as pd


class Formatter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod 
    def format(self, data):
        pass

    @abstractmethod
    def read_and_prepare(self, df, file):
        pass

    