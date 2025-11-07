class Interface:
    def input(self):
        raise NotImplementedError
    def output(self, move: str):
        raise NotImplementedError
class CompetitionInterface(Interface):
    def input(self):
        return input()
    def output(self, move: str):
        print(move)
class TestInterface(Interface):
    def input(self):
        return input("Enter move (SAN): ")
    def output(self, move: str):
        print(move)