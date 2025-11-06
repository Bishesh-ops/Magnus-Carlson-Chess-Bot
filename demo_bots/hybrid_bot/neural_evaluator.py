"""
Neural Network Evaluator - The "Intuitive Evaluator" / "Neuro-Aesthetic"

This module implements a compact neural network that perceives board positions
and assigns them a quantitative value. It's designed for speed - a "reflex" not a "meditation".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np


class BoardEncoder:
    """Encodes chess board positions into neural network input tensors"""
    
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
    """
    Compact Convolutional Neural Network for chess position evaluation.
    
    Architecture optimized for speed:
    - Small number of filters
    - Residual connections for gradient flow
    - Minimal depth for fast inference
    """
    
    def __init__(self, num_filters=64):
        super(CompactChessNet, self).__init__()
        
        # Initial convolution to process board state
        self.conv1 = nn.Conv2d(14, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Residual blocks for deep feature extraction
        self.res_block1 = self._make_residual_block(num_filters)
        self.res_block2 = self._make_residual_block(num_filters)
        self.res_block3 = self._make_residual_block(num_filters)
        
        # Value head - predicts position evaluation
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
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
        identity = x
        x = self.res_block1(x)
        x = F.relu(x + identity)
        
        identity = x
        x = self.res_block2(x)
        x = F.relu(x + identity)
        
        identity = x
        x = self.res_block3(x)
        x = F.relu(x + identity)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 32 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Output between -1 and 1
        
        return v


class NeuralEvaluator:
    """
    The Intuitive Evaluator - wraps the neural network for easy evaluation.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model weights (optional)
            device: torch device (cuda/cpu). Auto-detect if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = CompactChessNet(num_filters=64).to(self.device)
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
        """
        Evaluate a chess position.
        
        Args:
            board: chess.Board object
            
        Returns:
            float: Position evaluation from white's perspective
                   Positive values favor white, negative favor black
                   Range: approximately -1.0 to +1.0
        """
        # Encode board
        board_tensor = self.encoder.encode_board(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        # Get evaluation with inference optimizations
        with torch.no_grad():
            with torch.inference_mode():  # Extra speed for inference
                evaluation = self.model(board_tensor)
        
        return evaluation.item()
    
    def evaluate_batch(self, boards: list) -> list:
        """
        Evaluate multiple positions efficiently in a batch.
        
        Args:
            boards: List of chess.Board objects
            
        Returns:
            List of float evaluations
        """
        # Encode all boards
        board_tensors = [self.encoder.encode_board(board) for board in boards]
        board_tensors = np.stack(board_tensors)
        board_tensors = torch.from_numpy(board_tensors).to(self.device)
        
        # Get evaluations
        with torch.no_grad():
            evaluations = self.model(board_tensors)
        
        return evaluations.cpu().numpy().flatten().tolist()
    
    def train_mode(self):
        """Switch model to training mode"""
        self.model.train()
    
    def eval_mode(self):
        """Switch model to evaluation mode"""
        self.model.eval()
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
