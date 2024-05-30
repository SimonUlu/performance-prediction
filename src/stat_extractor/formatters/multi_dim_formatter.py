from contracts.formatter import Formatter

class MultiDimFormatter(Formatter):
    def __init__(self, file):
        super().__init__(file)


    def format(self):
        pass

    def read_and_prepare(self, df, file):
        return super().read_and_prepare(file)