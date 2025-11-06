import sys
from .play import play
from .train import train
from .interface import TestInterface, CompetitionInterface

if sys.argv[1] == "play":
    ### do stuff

    play(CompetitionInterface(), color = sys.argv[2])

elif sys.argv[1] == "train":
    ### do stuff

    train()

elif sys.argv[1] == "test":

    play(TestInterface(), color = sys.argv[2])

else:

    raise ValueError("Invalid argument recieved - 'play' or 'train' expected")