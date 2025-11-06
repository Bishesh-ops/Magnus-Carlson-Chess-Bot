import chess

class Interface:
    def __init__(self):
        pass

    def input():
        pass

    def output():
        pass

class CompetitionInterface(Interface):
    def __init__(self):
        super().__init__()

    def input(self):
        return input()

    def output(self, move):
        print(move)

class TestInterface(Interface):
    def __init__(self):
        super().__init__()

    def input(self):
        return input("Enter move (SAN): ")

    def output(self, move):
        print(move)