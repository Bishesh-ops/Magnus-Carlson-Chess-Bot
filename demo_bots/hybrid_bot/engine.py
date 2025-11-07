import chess
import time
import os
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .neural_evaluator import NeuralEvaluator
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}
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
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            openings_path = os.path.join(base_dir, "shared_resources", "openings.csv")
        if os.path.exists(openings_path):
            self._load_openings(openings_path)
    def _load_openings(self, path):
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if 'moves_list' in row and pd.notna(row['moves_list']):
                    moves_str = row['moves_list']
                    moves_str = moves_str.strip("[]'\"")
                    moves = [m.strip().strip("'\"") for m in moves_str.split(",")]
                    board = chess.Board()
                    for i, move_str in enumerate(moves):
                        try:
                            fen_key = board.fen().split(' ')[0]
                            move_san = move_str.split('.')[-1].strip()
                            if move_san:
                                move = board.parse_san(move_san)
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
            moves = self.openings[fen_key]
            sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
            legal_moves_set = set(board.legal_moves)
            for move, win_rate in sorted_moves:
                if move in legal_moves_set:
                    return move
        return None
@dataclass
class TTEntry:
    score: float
    depth: int
    flag: str
    move: Optional[chess.Move]
    generation: int
class HybridEngine:
    def __init__(self, neural_evaluator: Optional[NeuralEvaluator] = None, 
                 use_opening_book=True, search_depth=5, verbose=False):
        self.evaluator = neural_evaluator
        self.use_neural = neural_evaluator is not None
        self.opening_book = OpeningBook() if use_opening_book else None
        self.base_depth = search_depth
        self.max_time = 280
        self.nodes_searched = 0
        self.start_time = 0
        self.verbose = verbose
        self.transposition_table = {}
        self.history_table = {}
        self.killer_moves = {}
        self.eval_cache = {}
        self.butterfly_history = {}
        self.null_move_reduction_base = 2
        self.enable_null_move = True
        self.pv_line = []
        self.tt_generation = 0
        self.tt_max_age = 3
        self.tt_max_size = 400_000
        self.pawn_hash_cache = {}
        self.panic_mode = False
        self._last_root_moves = []
        self.last_depth_nodes = None
        self.branching_factors = []
    def _tt_store(self, key, score: float, depth: int, flag: str, move: Optional[chess.Move]):
        """Store TT entry using replacement policy: replace if deeper or same depth & newer generation."""
        new_entry = TTEntry(score=score, depth=depth, flag=flag, move=move, generation=self.tt_generation)
        old = self.transposition_table.get(key)
        if isinstance(old, TTEntry):
            if depth > old.depth or (depth == old.depth and self.tt_generation >= old.generation):
                self.transposition_table[key] = new_entry
        else:
            self.transposition_table[key] = new_entry
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
        try:
            return board.transposition_key()
        except AttributeError:
            return board.fen()
    def _pure_material_count(self, board: chess.Board) -> float:
        """Pure material count without positional bonuses - ensures we value captures correctly"""
        if board.is_checkmate():
            return -999999 if board.turn == chess.WHITE else 999999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        return score / 100.0
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
        return score / 100.0
    def _phase(self, board: chess.Board) -> float:
        """Return game phase index in [0,1]; 0 = opening, 1 = endgame."""
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
        file_counts_white = [0]*8
        file_counts_black = [0]*8
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        for sq in white_pawns:
            file_counts_white[sq % 8] += 1
        for sq in black_pawns:
            file_counts_black[sq % 8] += 1
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
        white_board = board.copy(stack=False)
        white_board.turn = chess.WHITE
        black_board = board.copy(stack=False)
        black_board.turn = chess.BLACK
        white_moves = len(list(white_board.legal_moves))
        black_moves = len(list(black_board.legal_moves))
        return (white_moves - black_moves) * 0.01
    def _king_safety(self, board: chess.Board) -> float:
        score = 0
        phase = self._phase(board)
        for color in (chess.WHITE, chess.BLACK):
            king_sq = board.king(color)
            if king_sq is None:
                continue
            rank = king_sq // 8
            file = king_sq % 8
            starting_rank = 0 if color == chess.WHITE else 7
            if phase < 0.5:
                if rank != starting_rank:
                    penalty = 0.8 * (1.0 - phase)
                    score += (-penalty if color == chess.WHITE else penalty)
                if file < 2 or file > 5:
                    if rank != starting_rank:
                        penalty = 0.5 * (1.0 - phase)
                        score += (-penalty if color == chess.WHITE else penalty)
            shield_files = [file + df for df in (-1,0,1) if 0 <= file+df < 8]
            shield_rank = rank + (1 if color == chess.WHITE else -1)
            missing = 0
            if 0 <= shield_rank < 8:
                for f in shield_files:
                    sq = shield_rank*8 + f
                    p = board.piece_at(sq)
                    if not (p and p.piece_type == chess.PAWN and p.color == color):
                        missing += 1
            penalty = missing * 0.25 * (1.0 - phase * 0.5)
            if board.is_attacked_by(not color, king_sq):
                penalty += 0.35 * (1.0 - phase * 0.5)
            score += (-penalty if color == chess.WHITE else penalty)
        return score
    def _piece_activity(self, board: chess.Board) -> float:
        score = 0
        center = {chess.D4, chess.D5, chess.E4, chess.E5}
        phase = self._phase(board)
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            starting_rank = 0 if color == chess.WHITE else 7
            if phase < 0.3:
                knights_developed = sum(1 for sq in board.pieces(chess.KNIGHT, color) if sq // 8 != starting_rank)
                bishops_developed = sum(1 for sq in board.pieces(chess.BISHOP, color) if sq // 8 != starting_rank)
                score += 0.15 * knights_developed * sign
                score += 0.15 * bishops_developed * sign
            for sq in board.pieces(chess.KNIGHT, color):
                if sq in center:
                    score += 0.20 * sign
                rank = sq // 8
                if (color == chess.WHITE and rank == 2) or (color == chess.BLACK and rank == 5):
                    score += 0.10 * sign
            for sq in board.pieces(chess.BISHOP, color):
                file = sq % 8
                rank = sq // 8
                if file in (2,3,4,5) and rank in (2,3,4,5):
                    score += 0.10 * sign
            for sq in board.pieces(chess.ROOK, color):
                file = sq % 8
                file_has_own_pawn = any(p % 8 == file for p in board.pieces(chess.PAWN, color))
                file_has_enemy_pawn = any(p % 8 == file for p in board.pieces(chess.PAWN, not color))
                if not file_has_own_pawn and not file_has_enemy_pawn:
                    score += 0.10 * sign
                elif not file_has_own_pawn and file_has_enemy_pawn:
                    score += 0.05 * sign
            if board.fullmove_number < 10:
                for sq in board.pieces(chess.QUEEN, color):
                    if color == chess.WHITE and sq not in (chess.D1):
                        score -= 0.05 * sign
                    if color == chess.BLACK and sq not in (chess.D8):
                        score -= 0.05 * sign
        return score
    def _protection_bonus(self, board: chess.Board) -> float:
        """Reward configurations where pawns/knights/bishops protect our king and queen.
        Penalize if king/queen squares lack such defenders or are over-attacked by enemy minors/pawns.
        Scaled roughly in pawn units.
        """
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            enemy = not color
            ksq = board.king(color)
            if ksq is not None:
                defenders = board.attackers(color, ksq)
                pawns_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.PAWN)
                knights_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.KNIGHT)
                bishops_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.BISHOP)
                prot = 0.12 * pawns_def + 0.10 * knights_def + 0.10 * bishops_def
                attackers = board.attackers(enemy, ksq)
                pawn_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.PAWN)
                knight_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.KNIGHT)
                bishop_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.BISHOP)
                pressure = 0.06 * pawn_att + 0.05 * knight_att + 0.05 * bishop_att
                if (pawns_def + knights_def + bishops_def) == 0:
                    prot -= 0.20
                score += sign * (prot - pressure)
            q_sqs = list(board.pieces(chess.QUEEN, color))
            for qsq in q_sqs:
                defenders = board.attackers(color, qsq)
                pawns_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.PAWN)
                knights_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.KNIGHT)
                bishops_def = sum(1 for sq in defenders if (p := board.piece_at(sq)) and p.piece_type == chess.BISHOP)
                prot = 0.06 * pawns_def + 0.05 * knights_def + 0.05 * bishops_def
                attackers = board.attackers(enemy, qsq)
                pawn_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.PAWN)
                knight_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.KNIGHT)
                bishop_att = sum(1 for sq in attackers if (p := board.piece_at(sq)) and p.piece_type == chess.BISHOP)
                pressure = 0.04 * pawn_att + 0.03 * knight_att + 0.03 * bishop_att
                if (pawns_def + knights_def + bishops_def) == 0:
                    prot -= 0.08
                score += sign * (prot - pressure)
        return score
    def _endgame_bonus(self, board: chess.Board) -> float:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                score += 0.15 * sign
            king_sq = board.king(color)
            if king_sq is not None:
                rank = king_sq // 8
                if 2 <= rank <= 5:
                    score += 0.05 * sign
        return score
    def _classical_eval(self, board: chess.Board) -> float:
        material = self._material_evaluation(board)
        pawn_struct = self._pawn_structure_eval(board)
        mobility = self._mobility(board)
        king_safety = self._king_safety(board)
        activity = self._piece_activity(board)
        protection = self._protection_bonus(board)
        endgame = self._endgame_bonus(board)
        phase = self._phase(board)
        mg_score = material + pawn_struct + king_safety + activity + mobility + protection
        eg_score = material + pawn_struct + endgame + activity * 0.5 + mobility * 0.5 + protection * 0.8
        blended = (1 - phase) * mg_score + phase * eg_score
        return blended
    def evaluate_position(self, board: chess.Board) -> float:
        key = self._tt_key(board)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached
        if board.is_checkmate():
            val = -999999 if board.turn == chess.WHITE else 999999
            self.eval_cache[key] = val
            return val
        if board.is_stalemate() or board.is_insufficient_material():
            self.eval_cache[key] = 0.0
            return 0.0
        material_only = self._pure_material_count(board)
        classical = self._classical_eval(board)
        neural_val = None
        if self.use_neural and self.evaluator:
            try:
                neural_val = self.evaluator.evaluate(board)
            except:
                neural_val = None
        phase = self._phase(board)
        if neural_val is not None:
            if phase < 0.25:
                val = 0.25 * material_only + 0.60 * classical + 0.15 * neural_val
            elif phase < 0.65:
                val = 0.30 * material_only + 0.55 * classical + 0.15 * neural_val
            else:
                val = 0.40 * material_only + 0.45 * classical + 0.15 * neural_val
        else:
            val = 0.50 * material_only + 0.50 * classical
        self.eval_cache[key] = val
        return val
    def _order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int = 0) -> List[chess.Move]:
        tt_move = None
        tt_entry = self.transposition_table.get(self._tt_key(board))
        if isinstance(tt_entry, TTEntry) and tt_entry.move in moves:
            tt_move = tt_entry.move
        killers = self.killer_moves.get(depth, [])
        def move_score(move):
            score = 0
            if tt_move is not None and move == tt_move:
                score += 100000
            if move in killers and not board.is_capture(move):
                score += 50000 - (killers.index(move) * 100)
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    capture_bonus = 100 * PIECE_VALUES[captured.piece_type]
                    score += capture_bonus - PIECE_VALUES[attacker.piece_type]
                see_gain = self._see(board, move)
                score += 3000 if see_gain > 0 else (-1500 if see_gain < 0 else 0)
            board.push(move)
            if board.is_check():
                score += 7000
            if board.is_repetition(2):
                score -= 100000
            elif board.is_repetition(1):
                score -= 30000
            board.pop()
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                if abs(move.from_square - move.to_square) == 2:
                    score += 8000
                else:
                    if board.fullmove_number < 15:
                        score -= 20000
                    elif board.fullmove_number < 30:
                        score -= 10000
            if piece and board.fullmove_number < 12:
                starting_rank = 0 if piece.color == chess.WHITE else 7
                if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    from_rank = move.from_square // 8
                    to_rank = move.to_square // 8
                    if from_rank == starting_rank and to_rank != starting_rank:
                        score += 5000
            move_key = (move.from_square, move.to_square)
            score += self.history_table.get(move_key, 0)
            mover = board.piece_at(move.from_square)
            if mover is not None:
                bkey = (mover.color, mover.piece_type, move.to_square)
                score += self.butterfly_history.get(bkey, 0)
            if move.promotion:
                if move.promotion == chess.QUEEN:
                    score += 12000
                else:
                    score += 8000
            return score
        ordered = sorted(moves, key=move_score, reverse=True)
        return ordered
    def _see(self, board: chess.Board, move: chess.Move) -> int:
        """Static Exchange Evaluation for captures. Returns net gain in centipawns for side to move."""
        if not board.is_capture(move):
            return 0

        to_sq = move.to_square
        from_sq = move.from_square
        
        # Initial capture value
        victim = board.piece_at(to_sq)
        if victim:
            value = PIECE_VALUES[victim.piece_type]
        else: # En-passant
            value = PIECE_VALUES[chess.PAWN]

        attacker = board.piece_at(from_sq)
        if not attacker:
             return 0 # Should not happen

        # Simulate the capture
        board.push(move)
        
        # Recursively calculate SEE from the opponent's perspective
        value -= self._see_recursive(board, to_sq, -1)
        
        board.pop()
        
        return value

    def _see_recursive(self, board: chess.Board, to_sq: chess.Square, sign: int) -> int:
        """Helper for SEE that finds the best response for the current side."""
        side = board.turn
        
        # Find the least valuable attacker
        attackers = board.attackers(not side, to_sq)
        if not attackers:
            return 0

        best_attacker_sq = -1
        min_attacker_value = float('inf')

        for attacker_sq in attackers:
            attacker_piece = board.piece_at(attacker_sq)
            if attacker_piece:
                attacker_value = PIECE_VALUES[attacker_piece.piece_type]
                if attacker_value < min_attacker_value:
                    min_attacker_value = attacker_value
                    best_attacker_sq = attacker_sq
        
        if best_attacker_sq == -1:
            return 0

        # Make the capture with the least valuable piece
        move = chess.Move(best_attacker_sq, to_sq)
        
        # The value of the piece being captured now
        captured_piece = board.piece_at(to_sq)
        if not captured_piece:
             return 0 # Should not happen
        value = PIECE_VALUES[captured_piece.piece_type]

        board.push(move)
        
        # After this capture, the opponent will recapture.
        # The value is from our perspective, so we subtract the opponent's gain.
        value -= self._see_recursive(board, to_sq, -sign)
        
        board.pop()
        
        return value
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
        stand_pat = self.evaluate_position(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in board.legal_moves:
            is_cap = board.is_capture(move)
            gives_chk = board.gives_check(move)
            if not (is_cap or gives_chk):
                continue
            if gives_chk and not is_cap and (stand_pat + 0.20 <= alpha):
                continue
            if is_cap:
                captured = board.piece_at(move.to_square)
                if captured is not None:
                    cap_val = PIECE_VALUES[captured.piece_type] / 100.0
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
        if time.time() - self.start_time > self.max_time:
            return self.evaluate_position(board)
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
        if (self.enable_null_move and depth >= 2 and not board.is_check()
            and not self._only_kings_and_pawns(board)
            and self._has_non_pawn_material(board, board.turn)):
            R = self.null_move_reduction_base + (1 if depth > 6 else 0)
            board.push(chess.Move.null())
            score = -self._alpha_beta(board, depth - 1 - R, -beta, -beta + 1, not maximizing)
            board.pop()
            if score >= beta:
                return beta
        if depth == 0 or board.is_game_over():
            return self._quiescence_search(board, alpha, beta)
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves, depth)
        if maximizing:
            max_eval = float('-inf')
            best_move = None
            original_alpha, original_beta = alpha, beta
            for idx, move in enumerate(ordered_moves):
                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                is_quiet = (not is_capture and not gives_check and not move.promotion)
                board.push(move)
                use_lmr = (depth >= 3 and idx > 3 and is_quiet)
                if use_lmr:
                    reduced_score = self._alpha_beta(board, depth - 2, alpha, beta, False)
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
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    mover = board.piece_at(move.from_square)
                    if mover is not None:
                        bkey = (mover.color, mover.piece_type, move.to_square)
                        self.butterfly_history[bkey] = self.butterfly_history.get(bkey, 0) + depth * depth
                    if not board.is_capture(move):
                        km = self.killer_moves.get(depth, [])
                        if move in km:
                            km.remove(move)
                        km.insert(0, move)
                        self.killer_moves[depth] = km[:2]
                    break
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
                    move_key = (move.from_square, move.to_square)
                    self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    mover = board.piece_at(move.from_square)
                    if mover is not None:
                        bkey = (mover.color, mover.piece_type, move.to_square)
                        self.butterfly_history[bkey] = self.butterfly_history.get(bkey, 0) + depth * depth
                    if not board.is_capture(move):
                        km = self.killer_moves.get(depth, [])
                        if move in km:
                            km.remove(move)
                        km.insert(0, move)
                        self.killer_moves[depth] = km[:2]
                    break
            flag = 'EXACT'
            if min_eval <= original_alpha:
                flag = 'UPPER'
            elif min_eval >= original_beta:
                flag = 'LOWER'
            self._tt_store(key, score=min_eval, depth=depth, flag=flag, move=best_move)
            return min_eval
    def find_best_move(self, board: chess.Board, time_limit: Optional[float] = None) -> chess.Move:
        board = board.copy()
        if self.opening_book and board.fullmove_number <= 15:
            opening_move = self.opening_book.get_opening_move(board)
            if opening_move:
                if self.verbose:
                    print(f"Using opening book move: {board.san(opening_move)}")
                return opening_move
        self.start_time = time.time()
        if time_limit:
            self.max_time = time_limit
        self.nodes_searched = 0
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        prev_best = None
        prev_root_move = None
        prev2_root_move = None
        for current_depth in range(1, self.base_depth + 5):
            self.tt_generation += 1
            if time.time() - self.start_time > self.max_time * 0.9:
                break
            elapsed = time.time() - self.start_time
            remaining = max(0.0, self.max_time - elapsed)
            self.panic_mode = remaining < self.max_time * 0.15
            saved_null = self.enable_null_move
            if self.panic_mode:
                self.enable_null_move = False
            nodes_before_depth = self.nodes_searched
            depth_best_move = None
            if prev_best is not None:
                window = 0.30 if self.panic_mode else 0.50
                alpha = prev_best - window
                beta = prev_best + window
            else:
                alpha = float('-inf')
                beta = float('inf')
            legal_moves = list(board.legal_moves)
            ordered_moves = self._order_moves(board, legal_moves, current_depth)
            if self.pv_line:
                pv_root = self.pv_line[0]
                if pv_root in legal_moves and pv_root in ordered_moves:
                    try:
                        ordered_moves.remove(pv_root)
                        ordered_moves.insert(0, pv_root)
                    except ValueError:
                        pass
            root_second_best = None
            for move in ordered_moves:
                board.push(move)
                if board.turn == chess.WHITE:
                    move_value = self._alpha_beta(board, current_depth - 1, alpha, beta, True)
                else:
                    move_value = self._alpha_beta(board, current_depth - 1, alpha, beta, False)
                if board.is_repetition(2):
                    repetition_penalty = 4.5
                    if abs(move_value) < 3.5:
                        if board.turn == chess.WHITE:
                            move_value -= repetition_penalty
                        else:
                            move_value += repetition_penalty
                elif board.is_repetition(1):
                    repetition_penalty = 2.0
                    if abs(move_value) < 2.5:
                        if board.turn == chess.WHITE:
                            move_value -= repetition_penalty
                        else:
                            move_value += repetition_penalty
                board.pop()
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
                if time.time() - self.start_time > self.max_time * 0.9:
                    break
            if depth_best_move:
                if depth_best_move in board.legal_moves:
                    best_move = depth_best_move
                    prev_best = best_value
                else:
                    import sys
                    print(f"WARNING: depth_best_move {depth_best_move} not legal at depth {current_depth}", file=sys.stderr)
            if self.verbose:
                move_str = 'None'
                if best_move:
                    try:
                        if best_move in board.legal_moves:
                            move_str = board.san(best_move)
                        else:
                            move_str = f"ILLEGAL({best_move})"
                    except:
                        move_str = f"ERROR({best_move})"
                print(f"Depth {current_depth}: best_move={move_str}, "
                      f"eval={best_value:.2f}, nodes={self.nodes_searched}, time={elapsed:.2f}s")
            if abs(best_value) > 900000:
                break
            self.pv_line = self._extract_pv_line(board, current_depth)
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
            if (projected_time_next * 1.2) > remaining and best_move is not None:
                break
            if depth_best_move is not None:
                if prev_root_move == depth_best_move and prev2_root_move == depth_best_move and root_second_best is not None:
                    margin = abs(best_value - root_second_best)
                    if margin >= 0.35:
                        break
                prev2_root_move = prev_root_move
                prev_root_move = depth_best_move
            self.enable_null_move = saved_null
        self._tt_prune_old()
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
        if best_move and abs(best_value) < 4.0:
            board.push(best_move)
            is_rep_2 = board.is_repetition(2)
            is_rep_1 = board.is_repetition(1)
            board.pop()
            if is_rep_2 or (is_rep_1 and abs(best_value) < 2.5):
                legal_moves_list = list(board.legal_moves)
                found_alternative = False
                for alt_move in legal_moves_list[:15]:
                    if alt_move == best_move:
                        continue
                    board.push(alt_move)
                    alt_is_rep_2 = board.is_repetition(2)
                    alt_is_rep_1 = board.is_repetition(1)
                    board.pop()
                    if not alt_is_rep_2 and not alt_is_rep_1:
                        old_move = best_move
                        best_move = alt_move
                        found_alternative = True
                        if self.verbose:
                            print(f"AVOIDING REPETITION: {board.san(old_move)} → {board.san(alt_move)}")
                        break
                if not found_alternative and is_rep_2:
                    for alt_move in legal_moves_list[:15]:
                        if alt_move == best_move:
                            continue
                        board.push(alt_move)
                        alt_is_rep_2 = board.is_repetition(2)
                        board.pop()
                        if not alt_is_rep_2:
                            best_move = alt_move
                            if self.verbose:
                                print(f"AVOIDING THREEFOLD: {board.san(best_move)} → {board.san(alt_move)}")
                            break
        legal_moves_list = list(board.legal_moves)
        if not legal_moves_list:
            raise ValueError("No legal moves available!")
        if best_move not in legal_moves_list:
            import sys
            print(f"WARNING: Engine produced illegal move {best_move} in position {board.fen()}", file=sys.stderr)
            print(f"Legal moves are: {[board.san(m) for m in legal_moves_list[:5]]}", file=sys.stderr)
            best_move = legal_moves_list[0]
        if len(self.transposition_table) > self.tt_max_size * 2:
            self.transposition_table.clear()
        if len(self.eval_cache) > 200000:
            self.eval_cache.clear()
        return best_move
