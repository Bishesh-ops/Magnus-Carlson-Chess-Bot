"""
Predictive Core - The "Structural Logic" / "Skeleton"

This module implements the deterministic search algorithm with alpha-beta pruning,
opening book integration, and move ordering for efficient tree exploration.
"""

import chess
import time
import os
import pandas as pd
from typing import Optional, Tuple, List
from .neural_evaluator import NeuralEvaluator


# Traditional piece values for fallback evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables for positional evaluation (simplified)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

KING_MIDDLE_GAME_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]


class OpeningBook:
    """Manages the opening book database"""
    
    def __init__(self, openings_path=None):
        self.openings = {}
        self.loaded = False
        
        if openings_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            openings_path = os.path.join(base_dir, "shared_resources", "openings.csv")
        
        if os.path.exists(openings_path):
            self._load_openings(openings_path)
    
    def _load_openings(self, path):
        """Load opening book from CSV"""
        try:
            df = pd.read_csv(path)
            
            # Build a dictionary mapping position FEN to best move
            for _, row in df.iterrows():
                if 'moves_list' in row and pd.notna(row['moves_list']):
                    # Parse the moves list
                    moves_str = row['moves_list']
                    # Remove brackets and quotes, split by comma
                    moves_str = moves_str.strip("[]'\"")
                    moves = [m.strip().strip("'\"") for m in moves_str.split(",")]
                    
                    # Simulate the position
                    board = chess.Board()
                    for i, move_str in enumerate(moves):
                        try:
                            # Store this position and next move
                            fen_key = board.fen().split(' ')[0]  # Just the piece positions
                            
                            # Parse move (format: "1.e4" or "Nf6")
                            move_san = move_str.split('.')[-1].strip()
                            if move_san:
                                move = board.parse_san(move_san)
                                
                                # Weight by performance
                                win_rate = row.get('White_win%', 50.0) if board.turn == chess.WHITE else row.get('Black_win%', 50.0)
                                
                                if fen_key not in self.openings:
                                    self.openings[fen_key] = []
                                self.openings[fen_key].append((move, win_rate))
                                
                                board.push(move)
                        except:
                            break
            
            self.loaded = True
            # Silently loaded (no output during competition)
        except Exception as e:
            # Silently fail (no output during competition)
            pass
    
    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best opening move for this position"""
        if not self.loaded:
            return None
        
        fen_key = board.fen().split(' ')[0]
        if fen_key in self.openings:
            # Get the move with highest win rate
            moves = self.openings[fen_key]
            best_move = max(moves, key=lambda x: x[1])[0]
            
            # Verify move is still legal
            if best_move in board.legal_moves:
                return best_move
        
        return None


class HybridEngine:
    """
    The complete hybrid engine combining neural evaluation with alpha-beta search.
    """
    
    def __init__(self, neural_evaluator: Optional[NeuralEvaluator] = None, 
                 use_opening_book=True, search_depth=4, verbose=False):
        """
        Initialize the hybrid chess engine.
        
        Args:
            neural_evaluator: Neural network evaluator (optional, will use material count if None)
            use_opening_book: Whether to use the opening book
            search_depth: Default search depth
            verbose: Whether to print search information (disable during competition)
        """
        self.evaluator = neural_evaluator
        self.use_neural = neural_evaluator is not None
        
        # Opening book
        self.opening_book = OpeningBook() if use_opening_book else None
        
        # Search parameters
        self.base_depth = search_depth
        self.max_time = 280  # Leave buffer in 300 second limit
        self.nodes_searched = 0
        self.start_time = 0
        self.verbose = verbose
        
        # Transposition table for memoization
        self.transposition_table = {}
        
        # Move ordering history
        self.history_table = {}
    
    def _material_evaluation(self, board: chess.Board) -> float:
        """
        Fallback material-based evaluation with piece-square tables.
        Returns evaluation from White's perspective.
        """
        if board.is_checkmate():
            return -999999 if board.turn == chess.WHITE else 999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                
                # Add positional bonus
                if piece.piece_type == chess.PAWN:
                    pos_value = PAWN_TABLE[square if piece.color == chess.WHITE else (63 - square)]
                elif piece.piece_type == chess.KNIGHT:
                    pos_value = KNIGHT_TABLE[square if piece.color == chess.WHITE else (63 - square)]
                elif piece.piece_type == chess.BISHOP:
                    pos_value = BISHOP_TABLE[square if piece.color == chess.WHITE else (63 - square)]
                elif piece.piece_type == chess.KING:
                    pos_value = KING_MIDDLE_GAME_TABLE[square if piece.color == chess.WHITE else (63 - square)]
                else:
                    pos_value = 0
                
                total_value = value + pos_value
                score += total_value if piece.color == chess.WHITE else -total_value
        
        return score / 100.0  # Normalize to similar scale as neural net
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a position using neural network or fallback.
        Returns evaluation from White's perspective.
        """
        # Check for terminal positions first
        if board.is_checkmate():
            return -999999 if board.turn == chess.WHITE else 999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Use neural evaluator if available
        if self.use_neural and self.evaluator:
            try:
                return self.evaluator.evaluate(board)
            except:
                pass
        
        # Fallback to material evaluation
        return self._material_evaluation(board)
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning.
        Priority: captures, checks, then history heuristic.
        """
        def move_score(move):
            score = 0
            
            # Prioritize captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    score += 10 * PIECE_VALUES[captured.piece_type] - PIECE_VALUES[attacker.piece_type]
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            
            # History heuristic
            move_key = (move.from_square, move.to_square)
            score += self.history_table.get(move_key, 0)
            
            # Prioritize promotions
            if move.promotion:
                score += 8000
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """
        Quiescence search to avoid horizon effect.
        Only searches tactical moves (captures, checks).
        """
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only consider captures and checks
        for move in board.legal_moves:
            if not (board.is_capture(move) or board.gives_check(move)):
                continue
            
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, 
                    beta: float, maximizing: bool) -> float:
        """
        Alpha-beta pruning search algorithm.
        The heart of the "Predictive Core".
        """
        self.nodes_searched += 1
        
        # Check time limit
        if time.time() - self.start_time > self.max_time:
            return self.evaluate_position(board)
        
        # Check transposition table
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            cached_depth, cached_score = self.transposition_table[board_hash]
            if cached_depth >= depth:
                return cached_score
        
        # Terminal node or depth limit reached
        if depth == 0 or board.is_game_over():
            # Use quiescence search at leaf nodes
            return self._quiescence_search(board, alpha, beta)
        
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    # Update history heuristic for cutoff moves
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    break
            
            # Store in transposition table
            self.transposition_table[board_hash] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    # Update history heuristic for cutoff moves
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    break
            
            # Store in transposition table
            self.transposition_table[board_hash] = (depth, min_eval)
            return min_eval
    
    def find_best_move(self, board: chess.Board, time_limit: Optional[float] = None) -> chess.Move:
        """
        Find the best move using iterative deepening with time management.
        
        Args:
            board: Current board position
            time_limit: Maximum time to search (seconds)
            
        Returns:
            Best move found
        """
        # Check opening book first
        if self.opening_book and board.fullmove_number <= 15:
            opening_move = self.opening_book.get_opening_move(board)
            if opening_move:
                if self.verbose:
                    print(f"Using opening book move: {board.san(opening_move)}")
                return opening_move
        
        # Setup for search
        self.start_time = time.time()
        if time_limit:
            self.max_time = time_limit
        self.nodes_searched = 0
        
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        # Iterative deepening: start with shallow depth, increase gradually
        for current_depth in range(1, self.base_depth + 5):
            if time.time() - self.start_time > self.max_time * 0.9:
                break
            
            depth_best_move = None
            alpha = float('-inf')
            beta = float('inf')
            
            legal_moves = list(board.legal_moves)
            ordered_moves = self._order_moves(board, legal_moves)
            
            for move in ordered_moves:
                board.push(move)
                
                if board.turn == chess.WHITE:  # After move, it's White's turn, so we just played Black
                    # We're minimizing (Black)
                    move_value = self._alpha_beta(board, current_depth - 1, alpha, beta, True)
                else:  # After move, it's Black's turn, so we just played White
                    # We're maximizing (White)
                    move_value = self._alpha_beta(board, current_depth - 1, alpha, beta, False)
                
                board.pop()
                
                # Check if this move is better
                if board.turn == chess.WHITE:
                    if move_value > best_value:
                        best_value = move_value
                        depth_best_move = move
                    alpha = max(alpha, move_value)
                else:
                    if move_value < best_value:
                        best_value = move_value
                        depth_best_move = move
                    beta = min(beta, move_value)
                
                # Time check
                if time.time() - self.start_time > self.max_time * 0.9:
                    break
            
            if depth_best_move:
                best_move = depth_best_move
            
            if self.verbose:
                elapsed = time.time() - self.start_time
                print(f"Depth {current_depth}: best_move={board.san(best_move) if best_move else 'None'}, "
                      f"eval={best_value:.2f}, nodes={self.nodes_searched}, time={elapsed:.2f}s")
            
            # If we've found a mate, no need to search deeper
            if abs(best_value) > 900000:
                break
        
        # Fallback: if no move found, pick first legal move
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
        
        # CRITICAL SAFETY CHECK: Verify move is legal before returning
        if best_move not in board.legal_moves:
            # This should never happen, but safety first
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
        
        # Clear some cache to prevent memory bloat
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
        
        return best_move
