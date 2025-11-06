"""
Training script for the Hybrid Chess Bot

This trains the neural evaluator on real chess games from the database.
"""

import os
import sys
import chess
import chess.pgn
import io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .neural_evaluator import NeuralEvaluator, BoardEncoder, CompactChessNet


class ChessGameDataset(Dataset):
    """Dataset of chess positions from real games"""
    
    def __init__(self, games_csv_path, max_games=None):
        """
        Load chess games and extract positions with outcomes.
        
        Args:
            games_csv_path: Path to games.csv
            max_games: Maximum number of games to load (None = all)
        """
        self.positions = []
        self.outcomes = []
        self.encoder = BoardEncoder()
        
        print(f"Loading games from {games_csv_path}...")
        self._load_games(games_csv_path, max_games)
        print(f"Loaded {len(self.positions)} positions from games")
    
    def _load_games(self, csv_path, max_games):
        """Load games and extract positions"""
        try:
            df = pd.read_csv(csv_path, nrows=max_games)
            
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    print(f"Processing game {idx}/{len(df)}")
                
                # Get game outcome
                winner = str(row.get('winner', 'draw')).strip().lower()
                if winner == 'white':
                    game_outcome = 1.0
                elif winner == 'black':
                    game_outcome = -1.0
                else:
                    game_outcome = 0.0
                
                # Parse moves
                moves_str = row.get('moves', '')
                if not isinstance(moves_str, str) or not moves_str:
                    continue
                
                # Create board and play through game
                board = chess.Board()
                moves = moves_str.split()
                
                # Extract positions from the game
                # Weight positions by their distance from the end (later positions more important)
                num_moves = len(moves)
                for move_num, move_san in enumerate(moves):
                    try:
                        move = board.parse_san(move_san)
                        board.push(move)
                        
                        # Skip very early positions (opening theory)
                        if move_num < 10:
                            continue
                        
                        # Calculate position value based on outcome and whose turn
                        # Adjust outcome based on whose perspective
                        position_value = game_outcome if board.turn == chess.WHITE else -game_outcome
                        
                        # Weight positions closer to the end more heavily
                        weight = (move_num / num_moves) ** 2
                        
                        # Store position
                        self.positions.append(self.encoder.encode_board(board))
                        self.outcomes.append(position_value * 0.7 + position_value * weight * 0.3)
                        
                    except Exception as e:
                        break
        
        except Exception as e:
            print(f"Error loading games: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = torch.from_numpy(self.positions[idx]).float()
        outcome = torch.tensor([self.outcomes[idx]], dtype=torch.float32)
        return position, outcome


def train(epochs=10, batch_size=64, learning_rate=0.001, max_games=1000):
    """
    Train the neural evaluator.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_games: Maximum games to load from database
    """
    print("=" * 80)
    print("HYBRID CHESS BOT - TRAINING THE NEURAL EVALUATOR")
    print("=" * 80)
    
    # Check if PyTorch and dependencies are available
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\nWARNING: PyTorch not installed. Skipping neural network training.")
        print("The bot will use material evaluation as fallback.")
        print("To train the neural network, install: pip install torch numpy pandas")
        return
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Find games database
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    games_path = os.path.join(base_dir, "shared_resources", "games.csv")
    
    if not os.path.exists(games_path):
        print(f"\nWARNING: Games database not found at {games_path}")
        print("Neural network training skipped. Bot will use material evaluation.")
        return
    
    # Load dataset
    try:
        dataset = ChessGameDataset(games_path, max_games=max_games)
        
        if len(dataset) == 0:
            print("No positions loaded. Training aborted.")
            return
        
        # Split into train/validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nTraining set: {train_size} positions")
        print(f"Validation set: {val_size} positions")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize model
    model = CompactChessNet(num_filters=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (positions, outcomes) in enumerate(train_loader):
            positions = positions.to(device)
            outcomes = outcomes.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(positions)
            loss = criterion(predictions, outcomes)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for positions, outcomes in val_loader:
                positions = positions.to(device)
                outcomes = outcomes.to(device)
                predictions = model(positions)
                loss = criterion(predictions, outcomes)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model to {model_path}")
        
        print()
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    train()