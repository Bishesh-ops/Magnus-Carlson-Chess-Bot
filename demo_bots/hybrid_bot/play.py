 

import chess
import time
import os
from .interface import Interface
from .engine import HybridEngine


def play(interface: Interface, color="w"):
    board = chess.Board()

    # Engine init
    try:
        from .neural_evaluator import NeuralEvaluator
        model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
        if os.path.exists(model_path):
            evaluator = NeuralEvaluator(model_path=model_path)
            engine = HybridEngine(neural_evaluator=evaluator, use_opening_book=True, search_depth=4)
        else:
            engine = HybridEngine(neural_evaluator=None, use_opening_book=True, search_depth=5)
    except:
        engine = HybridEngine(neural_evaluator=None, use_opening_book=True, search_depth=5)

    total_budget = 280.0
    time_remaining = total_budget

    while not board.is_game_over():
        # Decide whose turn: if it's our color, we move; else we read opponent
        our_turn = (color == "w" and board.turn == chess.WHITE) or (color == "b" and board.turn == chess.BLACK)
        if not our_turn:
            # Await opponent move
            try:
                opp = interface.input().strip()
                board.push_san(opp)
            except Exception as e:
                print(f"Error processing opponent move: {e}")
                break
            continue

        # Time allocation
        phase = board.fullmove_number
        moves_to_go = max(12, 50 - phase)
        min_floor = 3.0
        if phase <= 10:
            base = 5.0
        elif phase <= 25:
            base = 8.0
        else:
            base = 10.0
        fair_share = max(min_floor, (time_remaining / moves_to_go) * 1.3)
        time_per_move = max(min_floor, min(base, fair_share, time_remaining * 0.5))

        # Search and play
        start = time.time()
        try:
            move = engine.find_best_move(board, time_limit=time_per_move)
            if move not in board.legal_moves:
                raise ValueError("Engine produced illegal move")
        except Exception:
            # Fallback first legal move
            ls = list(board.legal_moves)
            if not ls:
                break
            move = ls[0]

        san = board.san(move)
        interface.output(san)
        board.push(move)
        elapsed = time.time() - start
        time_remaining = max(0.0, time_remaining - elapsed)

    # Game finished
    # Optional: output result
    # interface.output(f"Result: {board.result()}")
