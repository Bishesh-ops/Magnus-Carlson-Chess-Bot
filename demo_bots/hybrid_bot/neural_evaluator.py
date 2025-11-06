 

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# Simple move indexing: from_square (0..63) * 64 + to_square (0..63) yields 4096 possible non-promotion moves.
# Promotion moves are rarer; we can approximate by adding 4 promotion piece channels offset.
POLICY_SIZE = 4096 + (64 * 4)  # coarse upper bound; sparse usage

def move_to_index(move: chess.Move) -> int:
    base = move.from_square * 64 + move.to_square
    if move.promotion:
        # Promotion piece types mapped: N=0,B=1,R=2,Q=3
        promo_map = {chess.KNIGHT:0, chess.BISHOP:1, chess.ROOK:2, chess.QUEEN:3}
        offset = 4096 + move.from_square*4 + promo_map.get(move.promotion, 3)
        return offset
    return base

def index_flip_horizontal(idx: int) -> int:
    # Flip files horizontally for augmentation. Decode then re-encode.
    if idx >= 4096:
        # Promotion bucket
        rel = idx - 4096
        from_sq = rel // 4
        promo_sub = rel % 4
        fr_rank, fr_file = divmod(from_sq, 8)
        fr_file_f = 7 - fr_file
        # We cannot guarantee target square; treat as unchanged (approx) for simplicity.
        new_from = fr_rank*8 + fr_file_f
        return 4096 + new_from*4 + promo_sub
    from_sq = idx // 64
    to_sq = idx % 64
    fr_rank, fr_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    fr_file_f = 7 - fr_file
    to_file_f = 7 - to_file
    new_from = fr_rank*8 + fr_file_f
    new_to = to_rank*8 + to_file_f
    return new_from*64 + new_to


class BoardEncoder:
    
    # Piece representation: 12 piece types (6 per color) + en passant + castling rights
    PIECE_TO_INDEX = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    @staticmethod
    def encode_board(board: chess.Board) -> np.ndarray:
        """
        Encode board state into a tensor representation.
        Returns: numpy array of shape (14, 8, 8)
        
        Channels:
        0-11: Piece positions (one-hot encoded)
        12: Castling rights and en passant
        13: Turn indicator (all 1s if white's turn, all 0s if black's turn)
        """
        # Initialize 14 8x8 planes
        encoding = np.zeros((14, 8, 8), dtype=np.float32)
        
        # Encode piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_idx = BoardEncoder.PIECE_TO_INDEX[(piece.piece_type, piece.color)]
                encoding[piece_idx, rank, file] = 1.0
        
        # Encode castling rights (channel 12)
        if board.has_kingside_castling_rights(chess.WHITE):
            encoding[12, 0, 7] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            encoding[12, 0, 0] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            encoding[12, 7, 7] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            encoding[12, 7, 0] = 1.0
        
        # Encode en passant square
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            encoding[12, rank, file] = 0.5
        
        # Encode turn (channel 13)
        if board.turn == chess.WHITE:
            encoding[13, :, :] = 1.0
        
        return encoding


class CompactChessNet(nn.Module):
    
    def __init__(self, num_filters=128, num_res_blocks=6, policy=True):
        super(CompactChessNet, self).__init__()
        
        # Initial convolution to process board state
        self.conv1 = nn.Conv2d(14, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.res_blocks = nn.ModuleList([self._make_residual_block(num_filters) for _ in range(num_res_blocks)])
        
        # Value head - predicts position evaluation
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 64)
        self.value_fc3 = nn.Linear(64, 1)

        # Optional policy head
        self.use_policy = policy
        if self.use_policy:
            self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(32)
            self.policy_fc1 = nn.Linear(32 * 8 * 8, 512)
            self.policy_fc2 = nn.Linear(512, POLICY_SIZE)
        
    def _make_residual_block(self, num_filters):
        """Create a residual block with two convolutions"""
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters)
        )
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks with skip connections
        for block in self.res_blocks:
            identity = x
            x = block(x)
            x = F.relu(x + identity)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 32 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))

        if self.use_policy:
            p = F.relu(self.policy_bn(self.policy_conv(identity)))  # use last identity (features) as base
            p = p.view(-1, 32 * 8 * 8)
            p = F.relu(self.policy_fc1(p))
            p = self.policy_fc2(p)
            return v, p
        return v


class NeuralEvaluator:
    
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = CompactChessNet(num_filters=128, num_res_blocks=6, policy=True).to(self.device)
        self.encoder = BoardEncoder()
        
        # Load trained weights if provided
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except FileNotFoundError:
                print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        # Set to evaluation mode by default
        self.model.eval()
    
    def evaluate(self, board: chess.Board) -> float:
        # Encode board
        board_tensor = self.encoder.encode_board(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        # Get evaluation with inference optimizations
        with torch.no_grad():
            with torch.inference_mode():
                out = self.model(board_tensor)
                if isinstance(out, tuple):
                    evaluation, _ = out
                else:
                    evaluation = out
        
        return evaluation.item()
    
    def evaluate_batch(self, boards: list) -> list:
        # Encode all boards
        board_tensors = [self.encoder.encode_board(board) for board in boards]
        board_tensors = np.stack(board_tensors)
        board_tensors = torch.from_numpy(board_tensors).to(self.device)
        
        # Get evaluations
        with torch.no_grad():
            out = self.model(board_tensors)
            if isinstance(out, tuple):
                evaluations, _ = out
            else:
                evaluations = out
        return evaluations.cpu().numpy().flatten().tolist()
    
    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
