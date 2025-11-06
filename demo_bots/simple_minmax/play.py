import chess
import sys
from .interface import Interface


# Piece values for evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board(board):
    """
    Evaluate the board based on material count.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    if board.is_checkmate():
        # If it's White's turn and checkmate, White lost (bad for White)
        # If it's Black's turn and checkmate, Black lost (good for White)
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning.
    Always evaluates from White's perspective.
    
    Args:
        board: chess.Board object
        depth: remaining depth to search
        alpha: best value for maximizer
        beta: best value for minimizer
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        Best evaluation score from White's perspective
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    """
    Find the best move for the current player.
    
    Args:
        board: chess.Board object
        depth: search depth
    
    Returns:
        Best move in UCI notation (e.g., 'e2e4')
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    if board.turn == chess.WHITE:
        # White wants to MAXIMIZE the score
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, best_value)
    else:
        # Black wants to MINIMIZE the score
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, best_value)
    
    return best_move

def play(interface: Interface, color = "w"):
    search_depth = 4  # Can be any positive number
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    # color = interface.input()

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        try:
            best_move = find_best_move(board, search_depth)
            
            # CRITICAL: Verify move is legal
            if best_move is None or best_move not in board.legal_moves:
                print(f"Error: Generated illegal move {best_move}", file=sys.stderr)
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                best_move = legal_moves[0]
            
            # Convert to SAN before output
            san = board.san(best_move)
            interface.output(san)
            board.push(best_move)

            move = interface.input()
            board.push_san(move)
        except Exception as e:
            print(f"Error in minmax bot: {e}", file=sys.stderr)
            break
