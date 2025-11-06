 

import chess
import time
import os
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
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
        except Exception as e:
            pass
    
    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
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


@dataclass
class TTEntry:
    score: float
    depth: int
    flag: str  # 'EXACT', 'LOWER', 'UPPER'
    move: Optional[chess.Move]
    generation: int  # for aging/replacement


class HybridEngine:
    
    def __init__(self, neural_evaluator: Optional[NeuralEvaluator] = None, 
                 use_opening_book=True, search_depth=4, verbose=False):
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
        
        # Transposition table for memoization (key -> TTEntry)
        self.transposition_table = {}
        
        # Move ordering history
        self.history_table = {}

        # Killer moves heuristic: keep up to 2 killers per depth
        self.killer_moves = {}

        # Evaluation cache to avoid repeated neural/material evals
        self.eval_cache = {}

        # Butterfly history: keyed by (color, piece_type, to_square)
        self.butterfly_history = {}

        # Parameters for pruning/reductions
        self.null_move_reduction_base = 2  # R base for null move pruning
        self.enable_null_move = True

        # Principal Variation storage (updated each iteration)
        self.pv_line = []  # PV line moves from root

        # Transposition table aging
        self.tt_generation = 0
        self.tt_max_age = 3
        self.tt_max_size = 400_000

        # Pawn structure cache (hash -> evaluation component)
        self.pawn_hash_cache = {}

        # Time management and stability
        self.panic_mode = False
        self._last_root_moves = []  # store recent root PV moves
        self.last_depth_nodes = None  # nodes at previous completed depth (for branching factor)
        self.branching_factors = []   # recent branching factor estimates

    def _tt_store(self, key, score: float, depth: int, flag: str, move: Optional[chess.Move]):
        """Store TT entry using replacement policy: replace if deeper or same depth & newer generation."""
        new_entry = TTEntry(score=score, depth=depth, flag=flag, move=move, generation=self.tt_generation)
        old = self.transposition_table.get(key)
        if isinstance(old, TTEntry):
            if depth > old.depth or (depth == old.depth and self.tt_generation >= old.generation):
                self.transposition_table[key] = new_entry
        else:
            self.transposition_table[key] = new_entry
        # Optional soft cap prune by size
        if len(self.transposition_table) > self.tt_max_size:
            self._tt_prune_old()

    def _tt_prune_old(self):
        """Prune entries older than allowed age to keep memory in check."""
        min_gen = self.tt_generation - self.tt_max_age
        if min_gen <= 0:
            return
        keys_to_delete = [k for k, v in self.transposition_table.items() if isinstance(v, TTEntry) and v.generation < min_gen]
        for k in keys_to_delete:
            del self.transposition_table[k]

    def _tt_key(self, board: chess.Board):
        """Return a fast transposition key; fallback to FEN if not available."""
        # python-chess provides a fast Zobrist hash via transposition_key()
        try:
            return board.transposition_key()
        except AttributeError:
            # Fallback: include full FEN to distinguish castling/ep/turn
            return board.fen()
    
    def _material_evaluation(self, board: chess.Board) -> float:
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

    # ---------------- Classical Evaluation Components -----------------

    def _phase(self, board: chess.Board) -> float:
        """Return game phase index in [0,1]; 0 = opening, 1 = endgame."""
        # Total material (excluding kings and pawns) guides phase.
        remaining = 0
        start_total = (2*PIECE_VALUES[chess.KNIGHT] + 2*PIECE_VALUES[chess.BISHOP] + 2*PIECE_VALUES[chess.ROOK] + 2*PIECE_VALUES[chess.QUEEN])
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            remaining += PIECE_VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
        phase = 1.0 - (remaining / start_total)
        return max(0.0, min(1.0, phase))

    def _pawn_key(self, board: chess.Board) -> int:
        """Create a simple hashable key for pawn structure (positions + side to move)."""
        white_pawns = tuple(sorted(board.pieces(chess.PAWN, chess.WHITE)))
        black_pawns = tuple(sorted(board.pieces(chess.PAWN, chess.BLACK)))
        return hash((white_pawns, black_pawns))

    def _pawn_structure_eval(self, board: chess.Board) -> float:
        key = self._pawn_key(board)
        cached = self.pawn_hash_cache.get(key)
        if cached is not None:
            return cached
        score = 0
        # Files occupancy for doubled/isolated evaluation
        file_counts_white = [0]*8
        file_counts_black = [0]*8
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        for sq in white_pawns:
            file_counts_white[sq % 8] += 1
        for sq in black_pawns:
            file_counts_black[sq % 8] += 1
        # Isolated / doubled
        for f in range(8):
            c = file_counts_white[f]
            if c > 1:
                score -= 0.10 * (c - 1)
            if c == 1:
                if (f == 0 or file_counts_white[f-1] == 0) and (f == 7 or file_counts_white[f+1] == 0):
                    score -= 0.15
        for f in range(8):
            c = file_counts_black[f]
            if c > 1:
                score += 0.10 * (c - 1)
            if c == 1:
                if (f == 0 or file_counts_black[f-1] == 0) and (f == 7 or file_counts_black[f+1] == 0):
                    score += 0.15
        # Passed pawns (simple: no opposing pawn in front on same/adjacent file)
        for sq in white_pawns:
            rank = sq // 8
            file = sq % 8
            blocked = False
            for r in range(rank+1, 8):
                for df in (-1,0,1):
                    nf = file+df
                    if 0 <= nf < 8:
                        if (r*8 + nf) in black_pawns:
                            blocked = True
                            break
                if blocked:
                    break
            if not blocked:
                # Rank-scaled bonus
                score += 0.20 + 0.05 * rank
        for sq in black_pawns:
            rank = sq // 8
            file = sq % 8
            blocked = False
            for r in range(rank-1, -1, -1):
                for df in (-1,0,1):
                    nf = file+df
                    if 0 <= nf < 8:
                        if (r*8 + nf) in white_pawns:
                            blocked = True
                            break
                if blocked:
                    break
            if not blocked:
                score -= 0.20 + 0.05 * (7-rank)
        self.pawn_hash_cache[key] = score
        return score

    def _mobility(self, board: chess.Board) -> float:
        # Approx mobility: legal move count difference scaled
        white_board = board.copy(stack=False)
        white_board.turn = chess.WHITE
        black_board = board.copy(stack=False)
        black_board.turn = chess.BLACK
        white_moves = len(list(white_board.legal_moves))
        black_moves = len(list(black_board.legal_moves))
        return (white_moves - black_moves) * 0.01

    def _king_safety(self, board: chess.Board) -> float:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            king_sq = board.king(color)
            if king_sq is None:
                continue
            rank = king_sq // 8
            file = king_sq % 8
            shield_files = [file + df for df in (-1,0,1) if 0 <= file+df < 8]
            shield_rank = rank + (1 if color == chess.WHITE else -1)
            missing = 0
            if 0 <= shield_rank < 8:
                for f in shield_files:
                    sq = shield_rank*8 + f
                    p = board.piece_at(sq)
                    if not (p and p.piece_type == chess.PAWN and p.color == color):
                        missing += 1
            penalty = missing * 0.10
            # Open file danger: enemy rooks/queens attacking king square
            if board.is_attacked_by(not color, king_sq):
                penalty += 0.15
            score += (-penalty if color == chess.WHITE else penalty)
        return score

    def _piece_activity(self, board: chess.Board) -> float:
        score = 0
        center = {chess.D4, chess.D5, chess.E4, chess.E5}
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            # Knights in center
            for sq in board.pieces(chess.KNIGHT, color):
                if sq in center:
                    score += 0.10 * sign
            # Bishops on long diagonals (rough: corners attack squares) -> reward if on c1/f1/c8/f8 or central diagonals
            for sq in board.pieces(chess.BISHOP, color):
                file = sq % 8
                rank = sq // 8
                if file in (2,5) or rank in (2,5):
                    score += 0.06 * sign
            # Rooks on open/semi-open files
            for sq in board.pieces(chess.ROOK, color):
                file = sq % 8
                file_has_own_pawn = any(p % 8 == file for p in board.pieces(chess.PAWN, color))
                file_has_enemy_pawn = any(p % 8 == file for p in board.pieces(chess.PAWN, not color))
                if not file_has_own_pawn and not file_has_enemy_pawn:
                    score += 0.10 * sign
                elif not file_has_own_pawn and file_has_enemy_pawn:
                    score += 0.05 * sign
            # Queen early development penalty (before move 10 if far from original file)
            if board.fullmove_number < 10:
                for sq in board.pieces(chess.QUEEN, color):
                    if color == chess.WHITE and sq not in (chess.D1):
                        score -= 0.05 * sign
                    if color == chess.BLACK and sq not in (chess.D8):
                        score -= 0.05 * sign
        return score

    def _endgame_bonus(self, board: chess.Board) -> float:
        # Simple heuristics: bishop pair, rook behind passed pawn (skip due to complexity simplified), king activity
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                score += 0.15 * sign
            king_sq = board.king(color)
            if king_sq is not None:
                rank = king_sq // 8
                # Encourage central king in endgame (ranks 2-5 for white, 2-5 for black)
                if 2 <= rank <= 5:
                    score += 0.05 * sign
        return score

    def _classical_eval(self, board: chess.Board) -> float:
        material = self._material_evaluation(board)
        pawn_struct = self._pawn_structure_eval(board)
        mobility = self._mobility(board)
        king_safety = self._king_safety(board)
        activity = self._piece_activity(board)
        endgame = self._endgame_bonus(board)
        phase = self._phase(board)
        # Tapered: weight some terms by phase
        mg_score = material + pawn_struct + king_safety + activity + mobility
        eg_score = material + pawn_struct + endgame + activity * 0.5 + mobility * 0.5
        blended = (1 - phase) * mg_score + phase * eg_score
        return blended
    
    def evaluate_position(self, board: chess.Board) -> float:
        # Cache by transposition key
        key = self._tt_key(board)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached
        # Check for terminal positions first
        if board.is_checkmate():
            val = -999999 if board.turn == chess.WHITE else 999999
            self.eval_cache[key] = val
            return val
        
        if board.is_stalemate() or board.is_insufficient_material():
            self.eval_cache[key] = 0.0
            return 0.0
        
        classical = self._classical_eval(board)
        neural_val = None
        if self.use_neural and self.evaluator:
            try:
                neural_val = self.evaluator.evaluate(board)
            except:
                neural_val = None
        phase = self._phase(board)
        if neural_val is not None:
            # Blending strategy: more NN mid-game, more classical early & late
            if phase < 0.25:
                alpha = 0.55
            elif phase < 0.65:
                alpha = 0.75
            else:
                alpha = 0.60
            val = alpha * neural_val + (1 - alpha) * classical
        else:
            val = classical
        self.eval_cache[key] = val
        return val
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int = 0) -> List[chess.Move]:
        # Prefer TT move if available
        tt_move = None
        tt_entry = self.transposition_table.get(self._tt_key(board))
        if isinstance(tt_entry, TTEntry) and tt_entry.move in moves:
            tt_move = tt_entry.move

        killers = self.killer_moves.get(depth, [])

        def move_score(move):
            score = 0

            # TT move gets highest priority
            if tt_move is not None and move == tt_move:
                score += 100000

            # Killer moves (only for non-captures)
            if move in killers and not board.is_capture(move):
                # Newer killer gets slightly higher score
                score += 50000 - (killers.index(move) * 100)
            
            # Prioritize captures using MVV-LVA and SEE
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    score += 10 * PIECE_VALUES[captured.piece_type] - PIECE_VALUES[attacker.piece_type]
                # Static Exchange Evaluation bonus/penalty
                see_gain = self._see(board, move)
                score += 1000 if see_gain > 0 else (-500 if see_gain < 0 else 0)
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            
            # History heuristic
            move_key = (move.from_square, move.to_square)
            score += self.history_table.get(move_key, 0)
            # Butterfly history
            mover = board.piece_at(move.from_square)
            if mover is not None:
                bkey = (mover.color, mover.piece_type, move.to_square)
                score += self.butterfly_history.get(bkey, 0)
            
            # Prioritize promotions
            if move.promotion:
                score += 8000
            
            return score
        
        ordered = sorted(moves, key=move_score, reverse=True)
        return ordered

    def _see(self, board: chess.Board, move: chess.Move) -> int:
        """Simple Static Exchange Evaluation for captures. Returns net gain in centipawns for side to move."""
        if not board.is_capture(move):
            return 0
        to_sq = move.to_square
        from_sq = move.from_square
        side = board.turn
        victim = board.piece_at(to_sq)
        attacker = board.piece_at(from_sq)
        if victim is None or attacker is None:
            return 0

        gain = []
        occ = set(sq for sq in chess.SQUARES if board.piece_at(sq))
        attackers_white = {sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE and board.is_attacked_by(chess.WHITE, to_sq)}
        # Fallback lightweight SEE: victim value - attacker value
        return PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]
    
    def _has_non_pawn_material(self, board: chess.Board, color: bool) -> bool:
        """Return True if the side has any non-pawn material (to avoid zugzwang issues)."""
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if board.pieces(pt, color):
                return True
        return False

    def _only_kings_and_pawns(self, board: chess.Board) -> bool:
        """True if the board has only kings and pawns for both sides (zugzwang-prone endgames)."""
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if board.pieces(pt, chess.WHITE) or board.pieces(pt, chess.BLACK):
                return False
        return True

    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only consider captures and (some) checks with delta pruning
        for move in board.legal_moves:
            is_cap = board.is_capture(move)
            gives_chk = board.gives_check(move)
            if not (is_cap or gives_chk):
                continue
            # Only include checking moves if they might raise alpha above stand_pat
            if gives_chk and not is_cap and (stand_pat + 0.20 <= alpha):
                continue
            # Delta pruning for bad captures
            if is_cap:
                captured = board.piece_at(move.to_square)
                if captured is not None:
                    cap_val = PIECE_VALUES[captured.piece_type] / 100.0
                    # prune hopeless captures (no SEE here to keep it cheap)
                    if stand_pat + cap_val + 0.10 < alpha:
                        continue
            
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha

    def _extract_pv_line(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Follow TT best moves to reconstruct a PV line up to max_depth."""
        pv_moves: List[chess.Move] = []
        temp_board = board.copy(stack=False)
        for _ in range(max_depth):
            entry = self.transposition_table.get(self._tt_key(temp_board))
            if not isinstance(entry, TTEntry) or entry.move is None:
                break
            mv = entry.move
            if mv not in temp_board.legal_moves:
                break
            pv_moves.append(mv)
            temp_board.push(mv)
        return pv_moves
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, 
                    beta: float, maximizing: bool) -> float:
        self.nodes_searched += 1
        
        # Check time limit
        if time.time() - self.start_time > self.max_time:
            return self.evaluate_position(board)
        
        # Check transposition table
        key = self._tt_key(board)
        tt = self.transposition_table.get(key)
        if isinstance(tt, TTEntry) and tt.depth >= depth:
            if tt.flag == 'EXACT':
                return tt.score
            elif tt.flag == 'LOWER':
                alpha = max(alpha, tt.score)
            elif tt.flag == 'UPPER':
                beta = min(beta, tt.score)
            if alpha >= beta:
                return tt.score
        
        # Null-move pruning
        if (self.enable_null_move and depth >= 2 and not board.is_check()
            and not self._only_kings_and_pawns(board)
            and self._has_non_pawn_material(board, board.turn)):
            # Choose reduction R based on remaining depth
            R = self.null_move_reduction_base + (1 if depth > 6 else 0)
            # Do a null move: pass the turn without making a move
            board.push(chess.Move.null())
            # Use a narrow window around -beta for efficiency (verification search)
            score = -self._alpha_beta(board, depth - 1 - R, -beta, -beta + 1, not maximizing)
            board.pop()
            if score >= beta:
                return beta

        # Terminal node or depth limit reached
        if depth == 0 or board.is_game_over():
            # Use quiescence search at leaf nodes
            return self._quiescence_search(board, alpha, beta)
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves, depth)
        
        if maximizing:
            max_eval = float('-inf')
            best_move = None
            original_alpha, original_beta = alpha, beta
            for idx, move in enumerate(ordered_moves):
                # Evaluate move characteristics before push
                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                is_quiet = (not is_capture and not gives_check and not move.promotion)
                board.push(move)
                # Late Move Reductions (LMR) for quiet, late moves
                use_lmr = (depth >= 3 and idx > 3 and is_quiet)
                if use_lmr:
                    reduced_score = self._alpha_beta(board, depth - 2, alpha, beta, False)
                    # Verification re-search only if promising and not in panic mode
                    if (reduced_score > alpha) and (not self.panic_mode):
                        eval_score = self._alpha_beta(board, depth - 1, alpha, beta, False)
                    else:
                        eval_score = reduced_score
                else:
                    eval_score = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    # Update history heuristic for cutoff moves
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    # Butterfly history update
                    mover = board.piece_at(move.from_square)
                    if mover is not None:
                        bkey = (mover.color, mover.piece_type, move.to_square)
                        self.butterfly_history[bkey] = self.butterfly_history.get(bkey, 0) + depth * depth
                    # Killer move update (non-captures)
                    if not board.is_capture(move):
                        km = self.killer_moves.get(depth, [])
                        if move in km:
                            km.remove(move)
                        km.insert(0, move)
                        self.killer_moves[depth] = km[:2]
                    break
            
            # Store in transposition table with bound info
            flag = 'EXACT'
            if max_eval <= original_alpha:
                flag = 'UPPER'
            elif max_eval >= original_beta:
                flag = 'LOWER'
            self._tt_store(key, score=max_eval, depth=depth, flag=flag, move=best_move)
            return max_eval
        else:
            min_eval = float('inf')
            best_move = None
            original_alpha, original_beta = alpha, beta
            for idx, move in enumerate(ordered_moves):
                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                is_quiet = (not is_capture and not gives_check and not move.promotion)
                board.push(move)
                # Late Move Reductions (LMR) for quiet, late moves
                use_lmr = (depth >= 3 and idx > 3 and is_quiet)
                if use_lmr:
                    reduced_score = self._alpha_beta(board, depth - 2, alpha, beta, True)
                    if (reduced_score < beta) and (not self.panic_mode):
                        eval_score = self._alpha_beta(board, depth - 1, alpha, beta, True)
                    else:
                        eval_score = reduced_score
                else:
                    eval_score = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    # Update history heuristic for cutoff moves
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    # Butterfly history update
                    mover = board.piece_at(move.from_square)
                    if mover is not None:
                        bkey = (mover.color, mover.piece_type, move.to_square)
                        self.butterfly_history[bkey] = self.butterfly_history.get(bkey, 0) + depth * depth
                    # Killer move update (non-captures)
                    if not board.is_capture(move):
                        km = self.killer_moves.get(depth, [])
                        if move in km:
                            km.remove(move)
                        km.insert(0, move)
                        self.killer_moves[depth] = km[:2]
                    break
            
            # Store in transposition table with bound info
            flag = 'EXACT'
            if min_eval <= original_alpha:
                flag = 'UPPER'
            elif min_eval >= original_beta:
                flag = 'LOWER'
            self._tt_store(key, score=min_eval, depth=depth, flag=flag, move=best_move)
            return min_eval
    
    def find_best_move(self, board: chess.Board, time_limit: Optional[float] = None) -> chess.Move:
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
        prev_best = None
        prev_root_move = None
        prev2_root_move = None
        
        # Iterative deepening: start with shallow depth, increase gradually
        for current_depth in range(1, self.base_depth + 5):
            # Each iterative deepening iteration advances TT generation
            self.tt_generation += 1
            if time.time() - self.start_time > self.max_time * 0.9:
                break
            # Update panic mode based on elapsed time (no global clocks available)
            elapsed = time.time() - self.start_time
            remaining = max(0.0, self.max_time - elapsed)
            self.panic_mode = remaining < self.max_time * 0.15
            # Temporarily disable null move pruning during panic to avoid deep fail-high loops
            saved_null = self.enable_null_move
            if self.panic_mode:
                self.enable_null_move = False

            # Record nodes before starting this depth
            nodes_before_depth = self.nodes_searched
            
            depth_best_move = None
            # Aspiration window around previous best to speed up
            if prev_best is not None:
                window = 0.30 if self.panic_mode else 0.50  # narrower when in panic
                alpha = prev_best - window
                beta = prev_best + window
            else:
                alpha = float('-inf')
                beta = float('inf')
            
            legal_moves = list(board.legal_moves)
            ordered_moves = self._order_moves(board, legal_moves, current_depth)
            # Seed with previous PV move at root if available
            if self.pv_line:
                pv_root = self.pv_line[0]
                if pv_root in ordered_moves:
                    try:
                        ordered_moves.remove(pv_root)
                        ordered_moves.insert(0, pv_root)
                    except ValueError:
                        pass
            
            root_second_best = None
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
                        if depth_best_move is not None:
                            root_second_best = best_value if root_second_best is None else max(root_second_best, best_value)
                        best_value = move_value
                        depth_best_move = move
                    else:
                        root_second_best = move_value if root_second_best is None else max(root_second_best, move_value)
                    alpha = max(alpha, move_value)
                else:
                    if move_value < best_value:
                        if depth_best_move is not None:
                            root_second_best = best_value if root_second_best is None else min(root_second_best, best_value)
                        best_value = move_value
                        depth_best_move = move
                    else:
                        root_second_best = move_value if root_second_best is None else min(root_second_best, move_value)
                    beta = min(beta, move_value)
                
                # Time check
                if time.time() - self.start_time > self.max_time * 0.9:
                    break
            
            if depth_best_move:
                best_move = depth_best_move
                prev_best = best_value
            
            if self.verbose:
                print(f"Depth {current_depth}: best_move={board.san(best_move) if best_move else 'None'}, "
                      f"eval={best_value:.2f}, nodes={self.nodes_searched}, time={elapsed:.2f}s")
            
            # If we've found a mate, no need to search deeper
            if abs(best_value) > 900000:
                break

            # Update PV line for next iteration
            self.pv_line = self._extract_pv_line(board, current_depth)

            # Dynamic depth/time management: estimate branching and projected cost of next depth
            nodes_this_depth = self.nodes_searched - nodes_before_depth
            if self.last_depth_nodes is not None and self.last_depth_nodes > 0:
                bf = nodes_this_depth / self.last_depth_nodes
                self.branching_factors.append(bf)
                if len(self.branching_factors) > 5:
                    self.branching_factors = self.branching_factors[-5:]
            self.last_depth_nodes = nodes_this_depth
            total_elapsed = time.time() - self.start_time
            remaining = max(0.0, self.max_time - total_elapsed)
            avg_bf = (sum(self.branching_factors[-3:]) / len(self.branching_factors[-3:])) if self.branching_factors[-3:] else 2.0
            time_per_node = (total_elapsed / self.nodes_searched) if self.nodes_searched > 0 else 0.0
            projected_nodes_next = max(1, nodes_this_depth) * max(1.5, avg_bf)
            projected_time_next = projected_nodes_next * time_per_node
            # Early stop if projected next depth likely exceeds remaining time and we have a move
            if (projected_time_next * 1.2) > remaining and best_move is not None:
                break

            # Move stability early cutoff: if root PV move unchanged for two iterations and margin comfortable
            if depth_best_move is not None:
                if prev_root_move == depth_best_move and prev2_root_move == depth_best_move and root_second_best is not None:
                    margin = abs(best_value - root_second_best)
                    if margin >= 0.35:  # about a third of a pawn
                        break
                prev2_root_move = prev_root_move
                prev_root_move = depth_best_move

            # Restore null-move setting for next iteration
            self.enable_null_move = saved_null

        # Generation-based TT prune at the end of move search
        self._tt_prune_old()
        
        # Fallback: if no move found, pick first legal move
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
        
        # Verify move is legal before returning
        if best_move not in board.legal_moves:
            # This should never happen, but safety first
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
        
        # Clear some caches to prevent memory bloat
        # TT is pruned by generation; as a safeguard, hard clear if still too large
        if len(self.transposition_table) > self.tt_max_size * 2:
            self.transposition_table.clear()
        if len(self.eval_cache) > 200000:
            self.eval_cache.clear()
        
        return best_move
