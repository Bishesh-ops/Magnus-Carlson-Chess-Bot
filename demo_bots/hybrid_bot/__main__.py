import sys
from .play import play
from .train import train
from .interface import TestInterface, CompetitionInterface

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m hybrid_bot [play|train|test] [w|b]")
    cmd = sys.argv[1]
    if cmd == "play":
        color = sys.argv[2] if len(sys.argv) > 2 else "w"
        play(CompetitionInterface(), color=color)
    elif cmd == "train":
        train()
    elif cmd == "test":
        color = sys.argv[2] if len(sys.argv) > 2 else "w"
        play(TestInterface(), color=color)
    else:
        raise ValueError("Invalid argument received - 'play' or 'train' expected")

if __name__ == "__main__":
    main()