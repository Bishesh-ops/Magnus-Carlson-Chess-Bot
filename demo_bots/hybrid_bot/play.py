 

import chess
import time
import os
from .interface import Interface
from .engine import HybridEngine


def play(interface: Interface, color="w"):
    # Initialize board
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    
    # Initialize engine with neural evaluator
    engine = None
    try:
        from .neural_evaluator import NeuralEvaluator
        
    # Load trained model if present
        model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
        if os.path.exists(model_path):
            # Silent load
            evaluator = NeuralEvaluator(model_path=model_path)
            # Use depth 4 with neural net (slower but smarter evaluation)
            engine = HybridEngine(neural_evaluator=evaluator, use_opening_book=True, search_depth=4)
        else:
            # Fallback deeper search
            engine = HybridEngine(neural_evaluator=None, use_opening_book=True, search_depth=5)
    except Exception as e:
        # Silently fall back to material evaluation with deeper search
        engine = HybridEngine(neural_evaluator=None, use_opening_book=True, search_depth=5)
    
    # If black, read opponent first move
    if color == "b":
        move = interface.input()
        board.push_san(move)
    
    move_count = 0
    total_budget = 280.0
    time_remaining = total_budget
    
    # Game loop
    while True:
        move_count += 1
        
    # Time budget per move
        phase = board.fullmove_number
        moves_to_go = max(12, 50 - phase)
        min_floor = 3.0
        if phase <= 10:
            base = 5.0
        elif phase <= 25:
            base = 8.0
        else:
            base = 10.0
        # Respect remaining time
        fair_share = max(min_floor, (time_remaining / moves_to_go) * 1.3)
        time_per_move = max(min_floor, min(base, fair_share, time_remaining * 0.5))
        
    # Search best move
        try:
            start = time.time()
            best_move = engine.find_best_move(board, time_limit=time_per_move)
            
            # Validate move
            if best_move not in board.legal_moves:
                # Invalid move returned! Use fallback
                raise ValueError(f"Engine returned illegal move: {best_move}")
            
            # Convert to SAN notation
            move_san = board.san(best_move)
            
            # Output
            interface.output(move_san)
            
            # Apply locally
            board.push(best_move)
            # Update remaining time by actual think time (best effort)
            elapsed = time.time() - start
            time_remaining = max(0.0, time_remaining - elapsed)
            
        except Exception as e:
            # Fallback: pick first legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                # No legal moves (game over)
                break
            
            fallback_move = legal_moves[0]
            interface.output(board.san(fallback_move))
            board.push(fallback_move)
        
    # Read opponent move
        try:
            opponent_move = interface.input()
            board.push_san(opponent_move)
        except Exception as e:
            print(f"Error processing opponent move: {e}")
            break
        
    # Termination check
        if board.is_game_over():
            break
