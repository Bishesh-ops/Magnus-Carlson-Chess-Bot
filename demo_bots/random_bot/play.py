import chess
import random
import sys
from .interface import Interface


def play(interface: Interface, color = "w"):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        try:
            all_moves = list(board.legal_moves)
            if not all_moves:
                break
            
            best_move = random.choice(all_moves)
            
            # CRITICAL: Verify move is legal
            if best_move not in board.legal_moves:
                print(f"Error: Generated illegal move {best_move}", file=sys.stderr)
                best_move = all_moves[0]
            
            # Convert to SAN before output
            san = board.san(best_move)
            interface.output(san)
            board.push(best_move)

            move = interface.input()
            board.push_san(move)
        except Exception as e:
            print(f"Error in random bot: {e}", file=sys.stderr)
            break
