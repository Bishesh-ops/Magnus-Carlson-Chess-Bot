import sys
from .play import play
from .train import train
import argparse
from .interface import TestInterface, CompetitionInterface

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m hybrid_bot [play|train|test] [w|b]")
    cmd = sys.argv[1]
    if cmd == "play":
        color = sys.argv[2] if len(sys.argv) > 2 else "w"
        play(CompetitionInterface(), color=color)
    elif cmd == "train":
        # Parse optional training args here to avoid requiring direct script call
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--epochs", type=int, default=12)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--max-games", type=int, default=20000)
        parser.add_argument("--min-elo", type=int, default=2200)
        args, _ = parser.parse_known_args(sys.argv[2:])
        mg = None if args.max_games <= 0 else args.max_games
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_games=mg,
            min_elo=args.min_elo,
        )
    elif cmd == "test":
        color = sys.argv[2] if len(sys.argv) > 2 else "w"
        play(TestInterface(), color=color)
    else:
        raise ValueError("Invalid argument received - 'play' or 'train' expected")

if __name__ == "__main__":
    main()