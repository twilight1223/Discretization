import sys
import os

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('default.txt')